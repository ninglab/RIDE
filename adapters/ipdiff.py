import os
import sys

import numpy as np
import torch

from adapters.base import ModelAdapter, ReferenceContext
from adapters._common_kernels import pos_log_kernel, type_eps_from_target, type_log_kernel
from adapters._scaffold import compute_generated_mask, generated_mask_from_fixed_pos_npz

BASE_MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'base_model', 'ipdiff')
if BASE_MODEL_ROOT not in sys.path:
    sys.path.insert(0, BASE_MODEL_ROOT)

import utils.transforms as trans                                    # noqa: E402
from datasets.pl_data import FOLLOW_BATCH                            # noqa: E402
from graphbap.bapnet import BAPNet                                   # noqa: E402
from models.molopt_score_model import (                              # noqa: E402
    ScorePosNet3D, center_pos, extract, index_to_log_onehot, log_sample_categorical,
)
from torch_geometric.data import Batch                               # noqa: E402
from torch_geometric.transforms import Compose                       # noqa: E402
from utils import reconstruct as ipdiff_reconstruct                  # noqa: E402
from ddpm_inversion_scaffold import build_shift_table, load_complex, to_real  # noqa: E402


class IPDiffAdapter(ModelAdapter):
    name = 'ipdiff'

    def __init__(self, atom_enc_mode: str = 'add_aromatic', pocket_radius: int = 10):
        self.model = None
        self.net_cond = None
        self.ckpt_config = None
        self.pf = None
        self.lf = None
        self.atom_enc_mode = atom_enc_mode
        self.pocket_radius = pocket_radius
        self.device = 'cpu'
        self._T = None

    @property
    def T(self) -> int:
        return self._T

    def alpha_bar(self, t):
        return self.model.alphas_cumprod[int(t)]

    def load_checkpoint(self, ckpt_path: str, device: str = 'cuda',
                        net_cond_ckpt_path: str = None) -> None:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.ckpt_config = ckpt['config']
        self.pf = trans.FeaturizeProteinAtom()
        self.lf = trans.FeaturizeLigandAtom(self.atom_enc_mode)
        self.model = ScorePosNet3D(self.ckpt_config.model, protein_atom_feature_dim=self.pf.feature_dim,
                                   ligand_atom_feature_dim=self.lf.feature_dim).to(device)
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.model.eval()
        net_cond_ckpt_path = net_cond_ckpt_path or os.path.join(
            os.path.dirname(os.path.abspath(ckpt_path)), 'ipnet.pt')
        self.net_cond = BAPNet(ckpt_path=net_cond_ckpt_path,
                               hidden_nf=self.ckpt_config.net_cond.hidden_dim).to(device).eval()
        self.model, self.net_cond = self.model.double(), self.net_cond.double()
        self.device = device
        self._T = self.model.num_timesteps

    def load_reference(self, pdb_path: str, sdf_path: str, fixed_pos_npz: str = None
                       ) -> ReferenceContext:
        data, ligand_dict = load_complex(pdb_path, sdf_path, Compose([self.pf, self.lf]),
                                         self.pocket_radius)

        if fixed_pos_npz:
            cur_pos = np.asarray(ligand_dict['pos'], dtype=np.float64)
            if ligand_dict.get('center', None) is not None:
                cur_pos = cur_pos + np.asarray(ligand_dict['center'], dtype=np.float64)
            gen_mask = generated_mask_from_fixed_pos_npz(fixed_pos_npz, cur_pos).to(self.device)
        else:
            gen_mask, _ = compute_generated_mask(pdb_path, sdf_path)
            gen_mask = gen_mask.to(self.device)

        batch = Batch.from_data_list([data], follow_batch=FOLLOW_BATCH).to(self.device)
        batch.protein_pos, batch.ligand_pos = batch.protein_pos.double(), batch.ligand_pos.double()
        batch.protein_atom_feature = batch.protein_atom_feature.double()
        bl, bp = batch.ligand_element_batch, batch.protein_element_batch

        pp_c, lp_c, offset = center_pos(batch.protein_pos, batch.ligand_pos, bp, bl,
                                        mode=self.ckpt_config.model.center_pos_mode)
        x0_v = batch.ligand_atom_feature_full
        k_s, hbap_l, hbap_p = build_shift_table(
            self.model, self.net_cond, lp_c, x0_v, pp_c, batch.protein_atom_feature,
            bl, bp, self.device)

        state = {'pos': lp_c.clone(), 'v': x0_v.clone()}
        extra = dict(complex_data=data, k_s=k_s, hbap_l=hbap_l, hbap_p=hbap_p, protein_pos=pp_c,
                    protein_v=batch.protein_atom_feature, batch_protein=bp, batch_ligand=bl,
                    x0_pos=lp_c, x0_v=x0_v, offset=offset)
        return ReferenceContext(state=state, generated_mask=gen_mask, extra=extra)

    def expand_batch(self, ctx: ReferenceContext, state, B: int):
        batch = Batch.from_data_list([ctx.extra['complex_data']] * B,
                                     follow_batch=FOLLOW_BATCH).to(self.device)
        batch.protein_pos, batch.ligand_pos = batch.protein_pos.double(), batch.ligand_pos.double()
        batch.protein_atom_feature = batch.protein_atom_feature.double()
        bl, bp = batch.ligand_element_batch, batch.protein_element_batch
        pp_c, lp_c, offset = center_pos(batch.protein_pos, batch.ligand_pos, bp, bl,
                                        mode=self.ckpt_config.model.center_pos_mode)
        x0_v = batch.ligand_atom_feature_full

        drift = float((lp_c[:ctx.extra['x0_pos'].shape[0]] - ctx.extra['x0_pos']).abs().max())
        if drift > 1e-4:
            raise ValueError(f'expand_batch centering drifted by {drift:.3e} from the '
                             'single-item load_reference -- centering convention mismatch')

        extra = dict(ctx.extra)
        extra.update(hbap_l=ctx.extra['hbap_l'].repeat(B, 1), hbap_p=ctx.extra['hbap_p'].repeat(B, 1),
                    protein_pos=pp_c, protein_v=batch.protein_atom_feature,
                    batch_protein=bp, batch_ligand=bl, x0_pos=lp_c, x0_v=x0_v, offset=offset)
        state_b = {'pos': state['pos'].repeat(B, 1), 'v': state['v'].repeat(B)}
        gen_mask_b = ctx.generated_mask.repeat(B)
        ctx_b = ReferenceContext(state=state_b, generated_mask=gen_mask_b, extra=extra)
        return ctx_b, state_b

    def split_batch(self, state, B: int) -> list:
        n = state['pos'].shape[0] // B
        return [{'pos': state['pos'][i * n:(i + 1) * n], 'v': state['v'][i * n:(i + 1) * n]}
               for i in range(B)]

    def stack_batch(self, ctx: ReferenceContext, states: list):
        B = len(states)
        ctx_b, _ = self.expand_batch(ctx, states[0], B)
        state_b = {'pos': torch.cat([s['pos'] for s in states], dim=0),
                  'v': torch.cat([s['v'] for s in states], dim=0)}
        return ReferenceContext(state=state_b, generated_mask=ctx_b.generated_mask,
                                extra=ctx_b.extra), state_b

    def _predict_x0(self, state, t: int, ctx: ReferenceContext):
        B = int(ctx.extra['batch_ligand'].max().item()) + 1
        z_real = to_real(state['pos'], ctx.extra['k_s'][int(t)].repeat(B, 1), ctx.generated_mask)
        tt = torch.full((B,), int(t), dtype=torch.long, device=self.device)
        with torch.no_grad():
            preds = self.model(
                protein_pos=ctx.extra['protein_pos'], protein_v=ctx.extra['protein_v'],
                batch_protein=ctx.extra['batch_protein'], init_ligand_pos=z_real,
                init_ligand_v=state['v'], batch_ligand=ctx.extra['batch_ligand'],
                time_step=tt.to(state['pos'].dtype),
                hbap_protein=ctx.extra['hbap_p'], hbap_ligand=ctx.extra['hbap_l'])
        return preds['pred_ligand_pos'], preds['pred_ligand_v'], tt

    def sample_eps(self, state, t: int, ctx: ReferenceContext):
        pos_noise = torch.randn_like(state['pos'])
        n = self.model.num_classes
        uniform = torch.rand(state['v'].shape[0], n, device=self.device, dtype=state['pos'].dtype)
        gumbel = -torch.log(-torch.log(uniform.clamp_min(1e-30)) + 1e-30)
        return {'pos': pos_noise, 'type': gumbel}

    def reverse_step(self, state, t: int, ctx: ReferenceContext, eps):
        gen = ctx.generated_mask
        x0_hat, v_logits, tt = self._predict_x0(state, t, ctx)
        batch = ctx.extra['batch_ligand']
        c0 = extract(self.model.posterior_mean_c0_coef, tt, batch)
        ct = extract(self.model.posterior_mean_ct_coef, tt, batch)
        sigma = (0.5 * extract(self.model.posterior_logvar, tt, batch)).exp()
        next_pos = c0 * x0_hat + ct * state['pos'] + sigma * eps['pos']
        next_pos = torch.where(gen.unsqueeze(-1), next_pos, ctx.extra['x0_pos'])

        log_v_recon = torch.log_softmax(v_logits, dim=-1)
        log_v_now = index_to_log_onehot(state['v'], self.model.num_classes)
        log_model_prob = self.model.q_v_posterior(log_v_recon, log_v_now, tt, batch)
        next_v = (eps['type'] + log_model_prob).argmax(dim=-1)
        next_v = torch.where(gen, next_v, ctx.extra['x0_v'])
        return {'pos': next_pos, 'v': next_v}, eps

    def ddim_forward_step(self, state, t: int, ctx: ReferenceContext):
        gen = ctx.generated_mask
        x0_hat, v_logits, tt = self._predict_x0(state, t, ctx)
        a_prev, a_cur = self.model.alphas_cumprod[int(t)], self.model.alphas_cumprod[int(t) + 1]
        if t == 0:
            eps_hat = torch.zeros_like(state['pos'])
        else:
            eps_hat = (state['pos'] - a_prev.sqrt() * x0_hat) / (1.0 - a_prev).clamp_min(1e-12).sqrt()
        next_pos = a_cur.sqrt() * x0_hat + (1.0 - a_cur).sqrt() * eps_hat
        next_pos = torch.where(gen.unsqueeze(-1), next_pos, ctx.extra['x0_pos'])

        B = int(ctx.extra['batch_ligand'].max().item()) + 1
        tt_next = torch.full((B,), int(t) + 1, dtype=torch.long, device=self.device)
        next_v = state['v'].clone()
        if self.discrete_forward_sample:
            log_v_t_gen = index_to_log_onehot(state['v'][gen], self.model.num_classes)
            log_q_v_t1_gen = self.model.q_v_pred_one_timestep(
                log_v_t_gen, tt_next, ctx.extra['batch_ligand'][gen])
            next_v[gen] = log_sample_categorical(log_q_v_t1_gen)
        else:
            pred_log_v0 = torch.log_softmax(v_logits, dim=-1)
            log_q_v_t = self.model.q_v_pred(pred_log_v0[gen], tt_next, ctx.extra['batch_ligand'][gen])
            next_v[gen] = log_q_v_t.argmax(dim=-1)
        next_v[~gen] = ctx.extra['x0_v'][~gen]
        return {'pos': next_pos, 'v': next_v}

    def infer_eps(self, state_t, state_prev_target, t: int, ctx: ReferenceContext):
        gen = ctx.generated_mask
        x0_hat, v_logits, tt = self._predict_x0(state_t, t, ctx)
        batch = ctx.extra['batch_ligand']
        c0 = extract(self.model.posterior_mean_c0_coef, tt, batch)
        ct = extract(self.model.posterior_mean_ct_coef, tt, batch)
        mean_tilde = c0 * x0_hat + ct * state_t['pos']
        sigma = (0.5 * extract(self.model.posterior_logvar, tt, batch)).exp()
        eps_pos = torch.zeros_like(state_t['pos'])
        if t > 0:
            eps_pos = (state_prev_target['pos'] - mean_tilde) / sigma.clamp_min(1e-12)
        eps_pos = torch.where(gen.unsqueeze(-1), eps_pos, torch.zeros_like(eps_pos))

        pred_log_v0 = torch.log_softmax(v_logits, dim=-1)
        log_cur_v = index_to_log_onehot(state_t['v'], self.model.num_classes)
        log_model_prob = self.model.q_v_posterior(pred_log_v0, log_cur_v, tt, batch)
        eps_type = type_eps_from_target(log_model_prob, state_prev_target['v'])
        return {'pos': eps_pos, 'type': eps_type}

    def forward_log_kernel(self, query_states: list, query_t: int, endpoint_states: list,
                           endpoint_t: int, ctx: ReferenceContext):
        query_pos = torch.stack([s['pos'] for s in query_states])
        endpoint_pos = torch.stack([s['pos'] for s in endpoint_states])
        query_v = torch.stack([s['v'] for s in query_states])
        endpoint_v = torch.stack([s['v'] for s in endpoint_states])
        ab_q = self.model.alphas_cumprod[int(query_t)]
        ab_e = self.model.alphas_cumprod[int(endpoint_t)]
        log_pos = pos_log_kernel(query_pos, endpoint_pos, ab_q, ab_e)
        lacv_q = self.model.log_alphas_cumprod_v[int(query_t)]
        lacv_e = self.model.log_alphas_cumprod_v[int(endpoint_t)]
        log_type = type_log_kernel(query_v, endpoint_v, lacv_q, lacv_e,
                                   'uniform', self.model.num_classes)
        return log_pos + log_type

    def decode(self, state, ctx: ReferenceContext):
        return self.decode_batch(state, ctx)[0]

    def decode_batch(self, state, ctx: ReferenceContext) -> list:
        offset, batch = ctx.extra['offset'], ctx.extra['batch_ligand']
        pos = state['pos'] + offset[batch] if torch.is_tensor(offset) else state['pos']
        xyz_all = pos.detach().cpu().numpy()
        v_all = state['v'].detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        B = int(batch.max().item()) + 1
        results = []
        for i in range(B):
            sel = batch_np == i
            atomic_nums = trans.get_atomic_number_from_index(
                v_all[sel], mode=self.atom_enc_mode)
            try:
                results.append(ipdiff_reconstruct.reconstruct_from_generated(xyz_all[sel], atomic_nums))
            except Exception as e:
                print(f'[decode_batch] mol {i} failed to reconstruct: {e}')
                results.append(None)
        return results
