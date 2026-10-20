import torch
from tqdm.auto import tqdm

from adapters.base import ModelAdapter, ReferenceContext
from ride.utils.trajectory import InversionPayload
from .window import Window


def _tile(x, B: int):
    return x.repeat(B, *([1] * (x.dim() - 1))) if torch.is_tensor(x) else x


def tile_eps(eps: dict, B: int) -> dict:
    return {k: _tile(v, B) for k, v in eps.items()}


def chunk_sizes(total: int, batch_size: int = None) -> list:
    if not batch_size or batch_size >= total:
        return [total]
    n_full = total // batch_size
    sizes = [batch_size] * n_full
    if total % batch_size:
        sizes.append(total % batch_size)
    return sizes


def _perturb_one(adapter: ModelAdapter, ctx: ReferenceContext,
                 inversion: InversionPayload, window: Window, capture_times, B: int, desc_label: str,
                 bar_name: str = None):
    state = inversion.states[window.t1]
    if B != 1:
        ctx, state = adapter.expand_batch(ctx, state, B)
    captures = {}
    if window.t1 in capture_times:
        captures[window.t1] = state
    for t in tqdm(range(window.t1, window.t2, -1),
                  desc=f'{bar_name or "perturb"}[{window.t1}->{window.t2}] {desc_label}'):
        eps = adapter.sample_eps(state, t, ctx)
        state, _ = adapter.reverse_step(state, t, ctx, eps)
        if t - 1 in capture_times:
            captures[t - 1] = state
    for t in tqdm(range(window.t2, 0, -1),
                  desc=f'{bar_name or "replay"}[{window.t2}->0] {desc_label}'):
        noise = inversion.noises[t] if B == 1 else tile_eps(inversion.noises[t], B)
        state, _ = adapter.reverse_step(state, t, ctx, noise)
        if t - 1 in capture_times:
            captures[t - 1] = state
    return ctx, state, captures


def perturb(adapter: ModelAdapter, ctx: ReferenceContext,
           inversion: InversionPayload, window: Window, capture_times=(), B: int = 1,
           label: str = 'B', batch_size: int = None, bar_name: str = None):
    sizes = chunk_sizes(B, batch_size)
    if len(sizes) == 1:
        return _perturb_one(adapter, ctx, inversion, window, capture_times, B, f'{label}={B}', bar_name)

    final_states, captured = [], {t: [] for t in capture_times}
    for i, b in enumerate(sizes):
        desc_label = f'{label}={B} chunk{i + 1}/{len(sizes)}(b={b})'
        _, state_b, captures_b = _perturb_one(adapter, ctx, inversion, window, capture_times, b, desc_label,
                                              bar_name)
        final_states.extend(adapter.split_batch(state_b, b))
        for t in capture_times:
            captured[t].extend(adapter.split_batch(captures_b[t], b))
    ctx_out, state_out = adapter.stack_batch(ctx, final_states)
    captures_out = {}
    for t in capture_times:
        _, captures_out[t] = adapter.stack_batch(ctx, captured[t])
    return ctx_out, state_out, captures_out
