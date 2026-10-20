import sys
import os

import numpy as np
import torch

_CONDITAR_BASE_MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         'base_model', 'conditar')


def _import_conditar_data():
    import importlib
    stash = {k: v for k, v in sys.modules.items() if k == 'utils' or k.startswith('utils.')}
    for k in stash:
        del sys.modules[k]
    sys.path.insert(0, _CONDITAR_BASE_MODEL_ROOT)
    try:
        return importlib.import_module('utils.data')
    finally:
        sys.path.remove(_CONDITAR_BASE_MODEL_ROOT)
        for k in [k for k in sys.modules if k == 'utils' or k.startswith('utils.')]:
            del sys.modules[k]
        sys.modules.update(stash)


def compute_generated_mask(protein_path: str, ligand_path: str):
    conditar_data = _import_conditar_data()
    mol = conditar_data.load_ligand_mol(ligand_path, removeHs=True)
    if mol is None:
        raise RuntimeError(f'load_ligand_mol returned None for {ligand_path}')
    mask = conditar_data.compute_decompdiff_scaffold_mask(protein_path, ligand_path, mol)
    mask = torch.as_tensor(np.asarray(mask, dtype=np.int64).astype(bool))
    pos = torch.as_tensor(np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64))
    return mask, pos


def match_generated_mask(mask: torch.Tensor, mask_pos, target_pos, tol: float = 1e-3) -> torch.Tensor:
    mask_pos = np.asarray(mask_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    mask = np.asarray(mask.detach().cpu().numpy() if torch.is_tensor(mask) else mask)
    if len(mask_pos) != len(target_pos):
        raise ValueError(f'atom count mismatch: computed={len(mask_pos)} target={len(target_pos)}')
    out = torch.zeros(len(target_pos), dtype=torch.bool)
    for k in range(len(mask_pos)):
        dist = np.linalg.norm(target_pos - mask_pos[k], axis=1)
        j = int(dist.argmin())
        if dist[j] > tol:
            raise ValueError(f'no target atom within {tol} of computed atom {k}')
        out[j] = bool(mask[k])
    return out


def generated_mask_from_fixed_pos_npz(npz_path: str, target_pos, tol: float = 1e-3) -> torch.Tensor:
    d = np.load(npz_path)
    fixed_pos = np.asarray(d['pos'], dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    mask = torch.ones(len(target_pos), dtype=torch.bool)
    for k in range(len(fixed_pos)):
        dist = np.linalg.norm(target_pos - fixed_pos[k], axis=1)
        j = int(dist.argmin())
        if dist[j] > tol:
            raise ValueError(f'no target atom within {tol} of fixed atom {k} ({npz_path})')
        mask[j] = False
    return mask
