import torch

from adapters.base import ModelAdapter, ReferenceContext


def compute_value(adapter: ModelAdapter, candidate_states: list, t: int,
                  endpoint_states: list, endpoint_t: int,
                  anchor_state, anchor_t: int, rewards: torch.Tensor,
                  ctx: ReferenceContext):
    log_q_anchor = adapter.forward_log_kernel([anchor_state], anchor_t, endpoint_states, endpoint_t, ctx)
    log_w = adapter.forward_log_kernel(candidate_states, t, endpoint_states, endpoint_t, ctx) - log_q_anchor
    weights = torch.softmax(log_w, dim=-1)
    return weights @ rewards.to(weights.dtype), weights.max(dim=-1).values
