from dataclasses import dataclass, field
import os

import torch


@dataclass
class InversionPayload:
    states: dict
    noises: dict
    meta: dict = field(default_factory=dict)

    def save(self, path):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        torch.save({'states': self.states, 'noises': self.noises, 'meta': self.meta}, path)

    @classmethod
    def load(cls, path, map_location='cpu'):
        d = torch.load(path, map_location=map_location)
        return cls(states=d['states'], noises=d['noises'], meta=d.get('meta', {}))

    def to(self, device):
        def move(state):
            return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in state.items()}
        self.states = {t: move(s) for t, s in self.states.items()}
        self.noises = {t: move(s) for t, s in self.noises.items()}
        return self
