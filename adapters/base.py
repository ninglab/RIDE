from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ReferenceContext:
    state: Any
    generated_mask: torch.Tensor
    extra: dict


class ModelAdapter(ABC):
    name: str
    discrete_forward_sample: bool = False

    @property
    @abstractmethod
    def T(self) -> int:
        pass

    @abstractmethod
    def alpha_bar(self, t) -> torch.Tensor:
        pass

    @abstractmethod
    def load_checkpoint(self, ckpt_path: str, device: str = 'cuda') -> None:
        pass

    @abstractmethod
    def load_reference(self, pdb_path: str, sdf_path: str, fixed_pos_npz: str = None) -> ReferenceContext:
        pass

    @abstractmethod
    def expand_batch(self, ctx: ReferenceContext, state, B: int):
        pass

    @abstractmethod
    def split_batch(self, state, B: int) -> list:
        pass

    @abstractmethod
    def stack_batch(self, ctx: ReferenceContext, states: list):
        pass

    @abstractmethod
    def sample_eps(self, state, t: int, ctx: ReferenceContext):
        pass

    @abstractmethod
    def reverse_step(self, state, t: int, ctx: ReferenceContext, eps):
        pass

    @abstractmethod
    def ddim_forward_step(self, state, t: int, ctx: ReferenceContext):
        pass

    @abstractmethod
    def infer_eps(self, state_t, state_prev_target, t: int, ctx: ReferenceContext):
        pass

    @abstractmethod
    def forward_log_kernel(self, query_states: list, query_t: int, endpoint_states: list,
                           endpoint_t: int, ctx: ReferenceContext) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, state, ctx: ReferenceContext):
        pass

    @abstractmethod
    def decode_batch(self, state, ctx: ReferenceContext) -> list:
        pass
