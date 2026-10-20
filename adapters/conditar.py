import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict

from adapters.base import ModelAdapter, ReferenceContext
from adapters._common_kernels import pos_log_kernel, type_eps_from_target, type_log_kernel
from adapters._scaffold import generated_mask_from_fixed_pos_npz

BASE_MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'base_model', 'conditar')
if BASE_MODEL_ROOT not in sys.path:
    sys.path.insert(0, BASE_MODEL_ROOT)

from models.molopt_score_model import (          # noqa: E402
    ScorePosNet3D, center_pos, index_to_log_onehot, log_sample_categorical,
)
import utils.transforms as conditar_trans            # noqa: E402
import utils.data as conditar_data                   # noqa: E402
import utils.reconstruct as conditar_reconstruct      # noqa: E402
from datasets import get_dataset                  # noqa: E402
from datasets.mol_data import FOLLOW_BATCH         # noqa: E402
from torch_geometric.data import Batch             # noqa: E402
from torch_geometric.transforms import Compose     # noqa: E402


_POCKET_CONFIG = {
    'data': {'hydrogen': False},
    'pocket': {
        'embedding_dim': 32, 'hidden_size': 128, 'output_size': 128, 'point_dim': 3,
        'pocket_layer_num': 3, 'layer_num': 3, 'num_k': 16, 'num_k_residue': 8,
        'loss_type_weight': 0.5, 'noise': 1.0, 'noise_type': 'uniform',
        'residue_center': 'mean',
    },
}


class ConditarAdapter(ModelAdapter):
    name = 'conditar'

    def __init__(self):
        self.model = None
        self.ckpt_config = None
        self.dataset_config = None
        self.ligand_featurizer = None
        self.transform = None
        self.device = 'cpu'
        self._T = None

    @property
    def T(self) -> int:
        return self._T

    def alpha_bar(self, t):
        return self.model.alphas_cumprod[int(t)]

    def load_checkpoint(self, ckpt_path: str, device: str = 'cuda',
                        pocket_ckpt_path: str = None) -> None:
        ckpt = torch.load(ckpt_path, map_location=device)
        self.ckpt_config = ckpt['config']
        pocket_ckpt_path = pocket_ckpt_path or os.path.join(
            os.path.dirname(os.path.abspath(ckpt_path)), 'pocket_ae.pt')
        self.dataset_config = EasyDict({
            **_POCKET_CONFIG,
            'model': {'checkpoint_pocket': pocket_ckpt_path},
        })
        ligand_atom_mode = (self.ckpt_config.data.transform.ligand_atom_mode
                            if 'transform' in self.ckpt_config.data else 'full')
        self.ligand_featurizer = conditar_trans.FeaturizeLigandAtom(ligand_atom_mode)
        self.transform = Compose([self.ligand_featurizer, conditar_trans.FeaturizeLigandBond()])
        self.model = ScorePosNet3D(
            self.ckpt_config,
            ligand_atom_feature_dim=self.ligand_featurizer.feature_dim,
            ligand_bond_feature_dim=len(conditar_data.BOND_TYPES),
        ).to(device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self.model = self.model.double()
        bad = [n for n, p in self.model.named_parameters() if p.dtype != torch.float64]
        if bad:
            raise RuntimeError(
                f'{len(bad)} parameters stayed non-float64 after .double(), e.g. '
                f'{bad[:5]}. They are not reachable from the module tree.')
        self.device = device
        self._T = self.model.num_timesteps

    def _build_batch(self, pocket_data_raw, ligand_data_raw, B: int):
        pocket_data_b = Batch.from_data_list(
            [pocket_data_raw] * B, follow_batch=list(FOLLOW_BATCH)).to(self.device)
        for k, v in pocket_data_b.items():
            if torch.is_tensor(v) and v.is_floating_point():
                pocket_data_b[k] = v.double()
        ligand_batch = Batch.from_data_list(
            [ligand_data_raw] * B, follow_batch=list(FOLLOW_BATCH)).to(self.device)
        batch_ligand = ligand_batch.ligand_element_batch
        x0_pos = ligand_batch.ligand_coordinate.double()
        x0_v = ligand_batch.ligand_atom_feature_full
        x0_pos_centered, offset = center_pos(x0_pos, batch_ligand, mode='center')
        return pocket_data_b, batch_ligand, x0_pos_centered, x0_v, offset

    def load_reference(self, pdb_path: str, sdf_path: str, fixed_pos_npz: str = None
                       ) -> ReferenceContext:
        dataset = get_dataset(config=self.dataset_config, name='test',
                              files=[(pdb_path, sdf_path)], ligand_transform=self.transform)
        data = dataset[0]

        if fixed_pos_npz:
            generated_mask = generated_mask_from_fixed_pos_npz(
                fixed_pos_npz, data['ligand'].ligand_coordinate.numpy()).to(self.device)
        else:
            mol = conditar_data.load_ligand_mol(sdf_path, removeHs=True)
            if mol is None:
                raise RuntimeError(f'load_ligand_mol returned None for {sdf_path}')
            scaffold_mask = conditar_data.compute_decompdiff_scaffold_mask(pdb_path, sdf_path, mol)
            generated_mask = torch.as_tensor(
                np.asarray(scaffold_mask, dtype=np.int64).astype(bool), device=self.device)

        pocket_data, batch_ligand, x0_pos_centered, x0_v, offset = self._build_batch(
            data['pocket'], data['ligand'], B=1)

        state = {'pos': x0_pos_centered.clone(), 'v': x0_v.clone()}
        extra = dict(pocket_data=pocket_data, ligand_data=data['ligand'],
                    batch_ligand=batch_ligand, offset=offset,
                    x0_pos_centered=x0_pos_centered, x0_v=x0_v,
                    num_classes_eff=self.model.num_classes + int(self.model.v_mode == 'tomask'))
        return ReferenceContext(state=state, generated_mask=generated_mask, extra=extra)

    def expand_batch(self, ctx: ReferenceContext, state, B: int):
        pocket_data_b, batch_ligand, x0_pos_centered, x0_v, offset = self._build_batch(
            ctx.extra['pocket_data'], ctx.extra['ligand_data'], B)

        pos_b = state['pos'].repeat(B, 1)
        v_b = state['v'].repeat(B)
        gen_mask_b = ctx.generated_mask.repeat(B)

        extra = dict(ctx.extra)
        extra.update(pocket_data=pocket_data_b, batch_ligand=batch_ligand, offset=offset,
                    x0_pos_centered=x0_pos_centered, x0_v=x0_v)
        state_b = {'pos': pos_b, 'v': v_b}
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
        t_tensor = torch.full((B,), int(t), dtype=torch.long, device=self.device)
        with torch.no_grad():
            preds = self.model(
                ligand_pos_perturbed=state['pos'],
                ligand_v_perturbed=state['v'],
                batch_ligand=ctx.extra['batch_ligand'],
                pocket_data=ctx.extra['pocket_data'],
                ligand_shape=None,
                mask_shape_emb=None,
                if_test=True,
                pred_bond=False,
                time_step=t_tensor.to(state['pos'].dtype),
            )
        pred_x0_pos = preds['pred_ligand_pos']
        pred_x0_v_logits = preds['pred_ligand_v']
        if self.model.v_mode == 'tomask':
            pred_x0_v_logits = pred_x0_v_logits.clone()
            pred_x0_v_logits[:, -1] = -1.e5
        return pred_x0_pos, pred_x0_v_logits

    def sample_eps(self, state, t: int, ctx: ReferenceContext):
        pos_noise = torch.randn_like(state['pos'])
        n = ctx.extra['num_classes_eff']
        uniform = torch.rand(state['v'].shape[0], n, device=self.device)
        gumbel = -torch.log(-torch.log(uniform.clamp_min(1e-30)) + 1e-30)
        return {'pos': pos_noise, 'type': gumbel}

    def reverse_step(self, state, t: int, ctx: ReferenceContext, eps):
        gen = ctx.generated_mask
        B = int(ctx.extra['batch_ligand'].max().item()) + 1
        t_tensor = torch.full((B,), int(t), dtype=torch.long, device=self.device)
        pred_x0_pos, pred_x0_v_logits = self._predict_x0(state, t, ctx)
        batch = ctx.extra['batch_ligand']

        pos_mean = self.model.q_pos_posterior(x0=pred_x0_pos, xt=state['pos'],
                                              t=t_tensor, batch=batch)
        pos_std = (0.5 * self.model.posterior_logvar[int(t)]).exp()
        next_pos = pos_mean + (0.0 if t == 0 else pos_std) * eps['pos']

        pred_log_v0 = F.log_softmax(pred_x0_v_logits, dim=-1)
        log_current_v = index_to_log_onehot(state['v'], ctx.extra['num_classes_eff'])
        log_model_prob = self.model.q_v_posterior(pred_log_v0, log_current_v, t_tensor, batch)
        next_v = log_sample_categorical(log_model_prob, gumbel_noise=eps['type'])

        next_pos[~gen] = ctx.extra['x0_pos_centered'][~gen]
        next_v[~gen] = ctx.extra['x0_v'][~gen]
        return {'pos': next_pos, 'v': next_v}, eps

    def ddim_forward_step(self, state, t: int, ctx: ReferenceContext):
        gen = ctx.generated_mask
        pred_x0_pos, pred_x0_v_logits = self._predict_x0(state, t, ctx)
        alpha_t = self.model.alphas_cumprod[int(t)]
        alpha_t1 = self.model.alphas_cumprod[int(t) + 1]
        if t == 0:
            eps_pos = torch.zeros_like(state['pos'][gen])
        else:
            eps_pos = ((state['pos'][gen] - alpha_t.sqrt() * pred_x0_pos[gen])
                      / (1.0 - alpha_t).sqrt().clamp_min(1e-12))
        next_pos = state['pos'].clone()
        next_pos[gen] = alpha_t1.sqrt() * pred_x0_pos[gen] + (1.0 - alpha_t1).sqrt() * eps_pos
        next_pos[~gen] = ctx.extra['x0_pos_centered'][~gen]

        B = int(ctx.extra['batch_ligand'].max().item()) + 1
        t1_tensor = torch.full((B,), int(t) + 1, dtype=torch.long, device=self.device)
        next_v = state['v'].clone()
        if self.discrete_forward_sample:
            log_v_t_gen = index_to_log_onehot(state['v'][gen], self.model.num_classes)
            log_q_v_t1_gen = self.model.q_v_pred_one_timestep(
                log_v_t_gen, t1_tensor, ctx.extra['batch_ligand'][gen])
            next_v[gen] = log_sample_categorical(log_q_v_t1_gen)
        else:
            pred_log_v0 = F.log_softmax(pred_x0_v_logits, dim=-1)
            log_q_v_t1_gen = self.model.q_v_pred(pred_log_v0[gen], t1_tensor,
                                                 ctx.extra['batch_ligand'][gen])
            next_v[gen] = log_q_v_t1_gen.argmax(dim=-1)
        next_v[~gen] = ctx.extra['x0_v'][~gen]
        return {'pos': next_pos, 'v': next_v}

    def infer_eps(self, state_t, state_prev_target, t: int, ctx: ReferenceContext):
        gen = ctx.generated_mask
        B = int(ctx.extra['batch_ligand'].max().item()) + 1
        t_tensor = torch.full((B,), int(t), dtype=torch.long, device=self.device)
        pred_x0_pos, pred_x0_v_logits = self._predict_x0(state_t, t, ctx)
        batch = ctx.extra['batch_ligand']

        pos_mean = self.model.q_pos_posterior(x0=pred_x0_pos, xt=state_t['pos'],
                                              t=t_tensor, batch=batch)
        pos_std = (0.5 * self.model.posterior_logvar[int(t)]).exp()
        eps_pos = torch.zeros_like(state_t['pos'])
        if t > 0:
            eps_pos[gen] = (state_prev_target['pos'][gen] - pos_mean[gen]) / pos_std.clamp_min(1e-12)

        n = ctx.extra['num_classes_eff']
        pred_log_v0 = F.log_softmax(pred_x0_v_logits, dim=-1)
        log_current_v = index_to_log_onehot(state_t['v'], n)
        log_model_prob = self.model.q_v_posterior(pred_log_v0, log_current_v, t_tensor, batch)
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
                                   self.model.v_mode, self.model.num_classes)
        return log_pos + log_type

    def decode(self, state, ctx: ReferenceContext):
        return self.decode_batch(state, ctx)[0]

    def decode_batch(self, state, ctx: ReferenceContext) -> list:
        offset, batch = ctx.extra['offset'], ctx.extra['batch_ligand']
        pos = state['pos'] + offset[batch]
        xyz_all = pos.detach().cpu().numpy()
        v_all = state['v'].detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        B = int(batch.max().item()) + 1
        results = []
        for i in range(B):
            sel = batch_np == i
            atomic_nums = conditar_trans.get_atomic_number_from_index(
                v_all[sel], mode=self.ligand_featurizer.mode)
            try:
                results.append(conditar_reconstruct.reconstruct_from_generated(xyz_all[sel], atomic_nums))
            except Exception as e:
                print(f'[decode_batch] mol {i} failed to reconstruct: {e}')
                results.append(None)
        return results
