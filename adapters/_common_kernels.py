import math

import torch


def pos_log_kernel(query_pos: torch.Tensor, endpoint_pos: torch.Tensor,
                   alpha_bar_query: torch.Tensor, alpha_bar_endpoint: torch.Tensor,
                   eps: float = 1e-12) -> torch.Tensor:
    ratio = (alpha_bar_query.clamp_min(eps) / alpha_bar_endpoint.clamp_min(eps)
            ).clamp(min=eps, max=1.0 - eps)
    sigma2 = (1.0 - ratio).clamp_min(eps)
    mean = ratio.sqrt() * endpoint_pos
    num_dims = query_pos.shape[-2] * query_pos.shape[-1]
    dist = (query_pos[:, None] - mean[None]).pow(2).sum(dim=(-2, -1)) / num_dims
    return -0.5 * dist / sigma2


def type_log_kernel(query_v: torch.Tensor, endpoint_v: torch.Tensor,
                    log_alpha_cumprod_v_query: torch.Tensor,
                    log_alpha_cumprod_v_endpoint: torch.Tensor,
                    v_mode: str, num_classes: int, eps: float = 1e-12) -> torch.Tensor:
    log_ratio = (log_alpha_cumprod_v_query - log_alpha_cumprod_v_endpoint).clamp(max=-eps)
    log_1m_ratio = torch.log((1.0 - log_ratio.exp()).clamp_min(eps))
    match = (query_v[:, None, :] == endpoint_v[None, :, :])
    if v_mode == 'uniform':
        log_same = torch.logaddexp(log_ratio, log_1m_ratio - math.log(num_classes))
        log_diff = log_1m_ratio - math.log(num_classes)
    elif v_mode == 'tomask':
        log_same = log_ratio
        log_diff = torch.tensor(-1e9, dtype=log_ratio.dtype, device=log_ratio.device)
    else:
        raise ValueError(f'unknown v_mode: {v_mode!r}')
    return torch.where(match, log_same, log_diff).double().mean(dim=-1)


def type_eps_from_target(log_model_prob: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
    chosen = target_v.unsqueeze(-1)
    chosen_logit = log_model_prob.gather(1, chosen).squeeze(1)
    onehot = torch.nn.functional.one_hot(target_v, num_classes=log_model_prob.shape[-1]).bool()
    max_other = log_model_prob.masked_fill(onehot, -1e9).max(dim=-1).values
    boost = (max_other - chosen_logit + 1e-3).clamp_min(0.0)
    eps_type = torch.zeros_like(log_model_prob)
    eps_type.scatter_(1, chosen, boost.unsqueeze(-1))
    return eps_type
