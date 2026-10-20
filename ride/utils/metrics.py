import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.QED import qed as _rdkit_qed

from .sascorer import compute_sa_score as _rdkit_sa


def is_disconnected(mol) -> bool:
    if mol is None:
        return True
    return '.' in Chem.MolToSmiles(mol)


def sim_2d(mol_a, mol_b):
    if mol_a is None or mol_b is None:
        return float('nan')
    fp_a, fp_b = Chem.RDKFingerprint(mol_a), Chem.RDKFingerprint(mol_b)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _gaussian_overlap(c1, c2, alpha, prefactor):
    import torch
    r2 = (torch.cdist(c1, c2) ** 2.0).T
    a_sum = 2 * alpha
    return torch.sum(
        np.pi ** 1.5 * (prefactor ** 2) * torch.exp(-(alpha / a_sum) * r2) / a_sum ** 1.5)


def sim_3d(mol_a, mol_b, alpha=0.81, prefactor=0.8):
    if mol_a is None or mol_b is None:
        return float('nan')
    if mol_a.GetNumConformers() == 0 or mol_b.GetNumConformers() == 0:
        return float('nan')
    try:
        import torch
        c1 = torch.as_tensor(mol_a.GetConformer().GetPositions(), dtype=torch.float32)
        c2 = torch.as_tensor(mol_b.GetConformer().GetPositions(), dtype=torch.float32)
        vaa = _gaussian_overlap(c1, c1, alpha, prefactor)
        vbb = _gaussian_overlap(c2, c2, alpha, prefactor)
        vab = _gaussian_overlap(c1, c2, alpha, prefactor)
        return float(vab / (vaa + vbb - vab))
    except Exception:
        return float('nan')


def qed(mol):
    if mol is None:
        return float('nan')
    try:
        return float(_rdkit_qed(mol))
    except Exception:
        return float('nan')


def sa(mol):
    if mol is None:
        return float('nan')
    try:
        return float(_rdkit_sa(mol))
    except Exception:
        return float('nan')
