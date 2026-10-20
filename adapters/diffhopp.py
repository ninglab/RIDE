import os
import sys

import torch

from adapters.base import ModelAdapter, ReferenceContext
from adapters._scaffold import compute_generated_mask, generated_mask_from_fixed_pos_npz, match_generated_mask

BASE_MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'base_model', 'diffhopp')
if BASE_MODEL_ROOT not in sys.path:
    sys.path.insert(0, BASE_MODEL_ROOT)

from causal_analysis_scaffold import (        
    ScaffoldDiffusion, build_molecules_batched, com_free_randn, com_project,
    load_complex, load_model,
)


def _sq_dist(a, b):
    a2 = a.pow(2).sum(dim=1, keepdim=True)
    b2 = b.pow(2).sum(dim=1).unsqueeze(0)
    return (a2 - 2.0 * a.matmul(b.t()) + b2).clamp_min(0.0)


def _gaussian_log_density(query, mean, ratio, dof, eps: float = 1e-12) -> torch.Tensor:
    sigma2 = (1.0 - ratio).clamp_min(eps)
    q_flat = query.reshape(query.shape[0], -1)
    m_flat = mean.reshape(mean.shape[0], -1)
    dist = _sq_dist(q_flat, m_flat) / max(int(dof), 1)
    return -0.5 * dist / sigma2


class DiffHoppAdapter(ModelAdapter):
    name = 'diffhopp'

    def __init__(self, dtype=torch.float64, pos_weight: float = 1.0, type_weight: float = 1.0):
        self.lit_model = None
        self.dtype = dtype
        self.pos_weight = pos_weight
        self.type_weight = type_weight
        self.device = 'cpu'
        self._T = None

    @property
    def T(self) -> int:
        return self._T

    def alpha_bar(self, t):
        return self.lit_model.model.schedule.sqrt_alpha_bar[int(t)] ** 2

    def load_checkpoint(self, ckpt_path: str, device: str = 'cuda') -> None:
        self.lit_model = load_model(ckpt_path, device)
        self.device = device
        self._T = int(self.lit_model.model.T)

    def load_reference(self, pdb_path: str, sdf_path: str, fixed_pos_npz: str = None
                       ) -> ReferenceContext:
        data, _ligand = load_complex(pdb_path, sdf_path)
        target_pos = data['ligand'].pos.detach().cpu().numpy()

        if fixed_pos_npz:
            data['ligand'].scaffold_mask = generated_mask_from_fixed_pos_npz(fixed_pos_npz, target_pos)
        else:
            mask, mask_pos = compute_generated_mask(pdb_path, sdf_path)
            data['ligand'].scaffold_mask = match_generated_mask(mask, mask_pos, target_pos)
        ops = ScaffoldDiffusion(self.lit_model, data, self.device, dtype=self.dtype)
        pos, x = ops.initial_state(1)
        state = {'pos': pos, 'x': x}
        return ReferenceContext(state=state, generated_mask=ops.gen_mask, extra=dict(ops=ops))

    def expand_batch(self, ctx: ReferenceContext, state, B: int):
        ops = ctx.extra['ops']
        pos = state['pos'][0:1].repeat(B, 1, 1).clone()
        x = state['x'][0:1].repeat(B, 1, 1).clone()
        return ctx, {'pos': pos, 'x': x}

    def split_batch(self, state, B: int) -> list:
        return [{'pos': state['pos'][i:i + 1], 'x': state['x'][i:i + 1]} for i in range(B)]

    def stack_batch(self, ctx: ReferenceContext, states: list):
        pos_b = torch.cat([s['pos'] for s in states], dim=0)
        x_b = torch.cat([s['x'] for s in states], dim=0)
        return ctx, {'pos': pos_b, 'x': x_b}

    def sample_eps(self, state, t: int, ctx: ReferenceContext):
        ops = ctx.extra['ops']
        B = int(state['pos'].shape[0])
        eps_pos = com_free_randn((B, ops.num_gen, 3), device=self.device,
                                 dtype=state['pos'].dtype)
        eps_x = torch.randn(B, ops.num_gen, ops.num_features, device=self.device,
                            dtype=state['x'].dtype)
        return {'pos': eps_pos, 'x': eps_x}

    def reverse_step(self, state, t: int, ctx: ReferenceContext, eps):
        ops = ctx.extra['ops']
        mu_x, mu_pos, sigma = ops.p_mean_std(state['pos'], state['x'], t)
        scale = 0.0 if t == 0 else sigma
        next_pos, next_x = state['pos'].clone(), state['x'].clone()
        gen = ops.gen_mask
        next_x[:, gen] = mu_x + scale * eps['x']
        next_pos[:, gen] = com_project(mu_pos + scale * eps['pos'])
        return {'pos': next_pos, 'x': next_x}, eps

    def ddim_forward_step(self, state, t: int, ctx: ReferenceContext):
        ops = ctx.extra['ops']
        next_pos, next_x = ops.ddim_forward_step(state['pos'], state['x'],
                                                  t_scalar=t, t_next=int(t) + 1)
        return {'pos': next_pos, 'x': next_x}

    def infer_eps(self, state_t, state_prev_target, t: int, ctx: ReferenceContext):
        ops = ctx.extra['ops']
        mu_x, mu_pos, sigma = ops.p_mean_std(state_t['pos'], state_t['x'], t)
        gen = ops.gen_mask
        if t == 0:
            return {'pos': torch.zeros_like(mu_pos), 'x': torch.zeros_like(mu_x)}
        eps_x = (state_prev_target['x'][:, gen] - mu_x) / sigma
        eps_pos = (state_prev_target['pos'][:, gen] - mu_pos) / sigma
        return {'pos': eps_pos, 'x': eps_x}

    def forward_log_kernel(self, query_states: list, query_t: int, endpoint_states: list,
                           endpoint_t: int, ctx: ReferenceContext):
        ops = ctx.extra['ops']
        gen = ops.gen_mask
        ratio = (self.alpha_bar(query_t).clamp_min(1e-12) / self.alpha_bar(endpoint_t).clamp_min(1e-12)
                ).clamp(min=1e-12, max=1.0 - 1e-12)
        q_pos = torch.cat([s['pos'] for s in query_states], dim=0)[:, gen]
        q_x = torch.cat([s['x'] for s in query_states], dim=0)[:, gen]
        e_pos = torch.cat([s['pos'] for s in endpoint_states], dim=0)[:, gen]
        e_x = torch.cat([s['x'] for s in endpoint_states], dim=0)[:, gen]
        log_pos = _gaussian_log_density(q_pos, ratio.sqrt() * e_pos, ratio, ops.pos_dof)
        log_x = _gaussian_log_density(q_x, ratio.sqrt() * e_x, ratio, ops.x_dof)
        return self.pos_weight * log_pos + self.type_weight * log_x

    def decode(self, state, ctx: ReferenceContext):
        return self.decode_batch(state, ctx)[0]

    def decode_batch(self, state, ctx: ReferenceContext) -> list:
        ops = ctx.extra['ops']
        results = build_molecules_batched(ops, state['pos'], state['x'])
        return [mol if is_valid else None for mol, _smiles, is_valid, _is_complete in results]
