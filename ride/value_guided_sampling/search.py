import math

import torch
from tqdm.auto import tqdm

from adapters.base import ModelAdapter, ReferenceContext
from ride.utils.trajectory import InversionPayload
from ride.perturbation_segment_selection.window import Window
from ride.perturbation_segment_selection.perturb import perturb, tile_eps, chunk_sizes
from .kernel import compute_value
from .reward import RewardWeights, compute_reward


def run_batched(adapter, ctx, states: list, step_fn, batch_size: int = None):
    out, offset = [], 0
    for b in chunk_sizes(len(states), batch_size):
        chunk = states[offset:offset + b]
        offset += b
        ctx_b, state_b = adapter.stack_batch(ctx, chunk)
        state_b = step_fn(ctx_b, state_b, b)
        out.extend(adapter.split_batch(state_b, b))
    return out


def collect_endpoints(adapter: ModelAdapter, ctx: ReferenceContext,
                      inversion: InversionPayload, window: Window, reference_mol,
                      weights: RewardWeights, M: int = 100,
                      collect_full_perturb_num: int = None, branch_time: int = None,
                      batch_size: int = None):
    full_n = M if collect_full_perturb_num is None else min(collect_full_perturb_num, M)
    tail_n = M - full_n
    if tail_n > 0:
        if collect_full_perturb_num is None or collect_full_perturb_num <= 0:
            raise ValueError('tail branching requires collect_full_perturb_num > 0')
        if branch_time is None:
            raise ValueError('branch_time is required when collect_full_perturb_num < M')

    capture_times = (window.t2,) if branch_time is None else (branch_time, window.t2)
    ctx_full, state_full, captures = perturb(adapter, ctx, inversion, window, capture_times=capture_times,
                                             B=full_n, label='full_n', batch_size=batch_size,
                                             bar_name='collect')
    full_mols = adapter.decode_batch(state_full, ctx_full)
    endpoints = list(zip(adapter.split_batch(captures[window.t2], full_n),
                         (compute_reward(mol, reference_mol, weights) for mol in full_mols)))
    branch_states = (adapter.split_batch(captures[branch_time], full_n)
                    if branch_time is not None else None)

    if tail_n > 0:
        tail_start_states = [branch_states[i % full_n] for i in range(tail_n)]
        tail_states = run_batched(adapter, ctx, tail_start_states,
                                  lambda cb, sb, b: _run_reverse(adapter, cb, sb, branch_time, window.t2,
                                                                 desc=f'collect[{branch_time}->{window.t2}] b={b}'),
                                  batch_size)
        tail_final_states = run_batched(adapter, ctx, tail_states,
                                        lambda cb, sb, b: _replay_to_zero(adapter, cb, sb, inversion, window.t2, b,
                                                                          desc=f'replay[{window.t2}->0] b={b}'),
                                        batch_size)
        ctx_tail, state_tail = adapter.stack_batch(ctx, tail_final_states)
        tail_mols = adapter.decode_batch(state_tail, ctx_tail)
        endpoints += list(zip(tail_states, (compute_reward(mol, reference_mol, weights) for mol in tail_mols)))
    return endpoints, branch_states


def topk_with_parent_cap(values: torch.Tensor, parent_ids: list, k: int, parent_cap: int):
    order = torch.argsort(values, descending=True).tolist()
    counts, kept = {}, []
    for i in order:
        if counts.get(parent_ids[i], 0) >= parent_cap:
            continue
        kept.append(i)
        counts[parent_ids[i]] = counts.get(parent_ids[i], 0) + 1
        if len(kept) == k:
            break
    return kept


def _run_reverse(adapter, ctx, state, t_from, t_to, desc=None):
    steps = range(t_from, t_to, -1)
    if desc is not None:
        steps = tqdm(steps, desc=desc)
    for t in steps:
        eps = adapter.sample_eps(state, t, ctx)
        state, _ = adapter.reverse_step(state, t, ctx, eps)
    return state


def _replay_to_zero(adapter, ctx, state, inversion, t_from, b, desc=None):
    steps = range(t_from, 0, -1)
    if desc is not None:
        steps = tqdm(steps, desc=desc)
    for tt in steps:
        noise = inversion.noises[tt] if b == 1 else tile_eps(inversion.noises[tt], b)
        state, _ = adapter.reverse_step(state, tt, ctx, noise)
    return state


def value_guided_search(adapter: ModelAdapter, ctx: ReferenceContext,
                        inversion: InversionPayload, window: Window, reference_mol,
                        weights: RewardWeights, value_resample_start: int,
                        value_resample_end: int, resample_stride: int,
                        value_expand_children: int, value_parent_cap: int,
                        M: int = 100, num_sample: int = 1, batch_size: int = None):
    start_t = window.t2 + value_resample_start
    end_t = window.t2 + value_resample_end
    if not (0 < value_resample_end < value_resample_start <= window.L):
        raise ValueError(f'need 0 < value_resample_end({value_resample_end}) < '
                         f'value_resample_start({value_resample_start}) <= L({window.L})')
    if not (M > num_sample):
        raise ValueError(f'need M({M}) > num_sample({num_sample}): collect always branches '
                         f'num_sample prefixes at start_t out to M endpoints')

    endpoints, branch_states = collect_endpoints(adapter, ctx, inversion, window, reference_mol, weights,
                                                 M, collect_full_perturb_num=num_sample, branch_time=start_t,
                                                 batch_size=batch_size)
    endpoint_states = [s for s, _ in endpoints]
    endpoint_rewards = torch.tensor([r for _, r in endpoints], device=adapter.device)

    anchor_state = inversion.states[window.t1]
    particles = [(s, i) for i, s in enumerate(branch_states)]
    t = start_t
    n_resample_steps = math.ceil((start_t - end_t) / resample_stride)
    resample_bar = tqdm(total=n_resample_steps, desc=f'value_guided_search resample[{start_t}->{end_t}]')
    while t > end_t:
        t_next = max(t - resample_stride, end_t)
        expand_states, parent_ids = [], []
        for i, (state, _) in enumerate(particles):
            expand_states.extend([state] * value_expand_children)
            parent_ids.extend([i] * value_expand_children)
        children = run_batched(adapter, ctx, expand_states,
                               lambda cb, sb, b: _run_reverse(adapter, cb, sb, t, t_next),
                               batch_size)

        values, _top_weight = compute_value(adapter, children, t_next, endpoint_states,
                                            window.t2, anchor_state, window.t1,
                                            endpoint_rewards, ctx)
        keep = topk_with_parent_cap(values, parent_ids, num_sample, value_parent_cap)
        particles = [(children[i], parent_ids[i]) for i in keep]
        t = t_next
        resample_bar.update(1)
    resample_bar.close()

    def _survivor_tail(cb, sb, b):
        sb = _run_reverse(adapter, cb, sb, end_t, window.t2, desc=f'perturb[{end_t}->{window.t2}] b={b}')
        return _replay_to_zero(adapter, cb, sb, inversion, window.t2, b, desc=f'replay[{window.t2}->0] b={b}')
    survivor_states = [s for s, _ in particles]
    final_states = run_batched(adapter, ctx, survivor_states, _survivor_tail, batch_size)
    ctx_s, state_s = adapter.stack_batch(ctx, final_states)
    mols = adapter.decode_batch(state_s, ctx_s)
    survivors = [(mol, compute_reward(mol, reference_mol, weights)) for mol in mols]
    best_mol, best_reward = max(survivors, key=lambda p: p[1])
    return best_mol, best_reward, survivors
