from dataclasses import dataclass


@dataclass
class Window:
    t1: int
    L: int
    t2: int

    @property
    def has_tail(self) -> bool:
        return self.t2 > 0


def make_window(t1: int, L: int) -> Window:
    if L <= 0:
        raise ValueError(f'L must be positive, got {L}')
    if t1 - L < 0:
        raise ValueError(f't1-L must be >= 0, got t1={t1} L={L}')
    return Window(t1=t1, L=L, t2=t1 - L)
