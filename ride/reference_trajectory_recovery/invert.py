import torch
from tqdm.auto import tqdm

from adapters.base import ModelAdapter, ReferenceContext
from ride.utils.trajectory import InversionPayload


def _cast(state, dtype):
    def cast_one(v):
        if torch.is_tensor(v) and v.is_floating_point():
            return v.to(dtype)
        return v
    if isinstance(state, dict):
        return {k: cast_one(v) for k, v in state.items()}
    return cast_one(state)


def ddim_invert(adapter: ModelAdapter, ctx: ReferenceContext, dtype=torch.float64):
    state = _cast(ctx.state, dtype)
    states = {0: state}
    for t in tqdm(range(adapter.T - 1), desc='ddim_invert'):
        state = adapter.ddim_forward_step(state, t, ctx)
        states[t + 1] = state
    return states


def infer_ddpm_noise(adapter: ModelAdapter, ctx: ReferenceContext, states: dict):
    return {t: adapter.infer_eps(states[t], states[t - 1], t, ctx)
            for t in tqdm(range(adapter.T - 1, 0, -1), desc='infer_ddpm_noise')}


def recover_reference_trajectory(adapter: ModelAdapter, ctx: ReferenceContext,
                                 dtype=torch.float64) -> InversionPayload:
    states = ddim_invert(adapter, ctx, dtype=dtype)
    noises = infer_ddpm_noise(adapter, ctx, states)
    return InversionPayload(states=states, noises=noises, meta={'dtype': str(dtype)})
