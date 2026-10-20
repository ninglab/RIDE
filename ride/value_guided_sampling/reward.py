from dataclasses import dataclass

from ride.utils.metrics import sim_2d, sim_3d, qed, sa, is_disconnected


@dataclass
class RewardWeights:
    w_2d: float = 0.0
    w_3d: float = 1.0
    w_qed: float = 0.0
    w_sa: float = 0.0


def compute_reward(mol, reference_mol, weights: RewardWeights) -> float:
    if mol is None or is_disconnected(mol):
        return 0.0
    return (weights.w_2d * sim_2d(mol, reference_mol)
            + weights.w_3d * sim_3d(mol, reference_mol)
            + weights.w_qed * qed(mol)
            + weights.w_sa * sa(mol))
