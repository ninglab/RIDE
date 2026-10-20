import os
from typing import List

import numpy as np
import rdkit.Chem
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import nn as nn

atom_names = ["C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F"]

residue_names = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


def get_ligand_scaffold_mask(ligand: rdkit.Chem.Mol) -> torch.Tensor:
    murcko_scaffold = MurckoScaffold.GetScaffoldForMol(ligand)
    scaffold_substructures = ligand.GetSubstructMatches(murcko_scaffold)
    substructure_indexes = [
        i for substructure in scaffold_substructures for i in substructure
    ]
    mask = torch.zeros(len(ligand.GetAtoms()), dtype=torch.bool)
    mask[substructure_indexes] = True
    return mask


def resolve_scaffold_npz(npz_dir: str, identifier: str) -> str:
    """Locate the SBDDcode scaffold npz for a given complex identifier."""
    for name in (f"{identifier}_ligand.npz", f"{identifier}.npz"):
        path = os.path.join(npz_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"no scaffold npz for identifier '{identifier}' in {npz_dir} "
        f"(tried {identifier}_ligand.npz, {identifier}.npz)"
    )


def get_ligand_scaffold_mask_from_npz(
    ligand: rdkit.Chem.Mol, npz_path: str, tol: float = 1e-3
) -> torch.Tensor:
    """SBDDcode-style scaffold mask from a saved npz (DecompDiff extraction).

    The npz stores RAW heavy-atom positions + scaffold_mask (1 = scaffold -> to be
    regenerated, 0 = non-scaffold -> fixed). ``ligand`` here is the heavy-atom mol
    (H already removed) in the original SDF frame, so heavy atoms are matched to the
    npz by coordinate. Returns a bool mask with True on the scaffold (regenerated)
    atoms, matching the convention of ``get_ligand_scaffold_mask``.
    """
    d = np.load(npz_path)
    heavy_pos = np.asarray(d["heavy_pos"], dtype=np.float64)
    scaffold_mask = np.asarray(d["scaffold_mask"], dtype=np.int32)
    lig_pos = np.asarray(ligand.GetConformer().GetPositions(), dtype=np.float64)

    if len(heavy_pos) != ligand.GetNumAtoms():
        raise ValueError(
            f"heavy-atom count mismatch: npz={len(heavy_pos)} "
            f"ligand={ligand.GetNumAtoms()} ({npz_path})"
        )

    mask = torch.zeros(ligand.GetNumAtoms(), dtype=torch.bool)
    for k in range(len(heavy_pos)):
        dist = np.linalg.norm(lig_pos - heavy_pos[k], axis=1)
        j = int(dist.argmin())
        if dist[j] >= tol:
            raise ValueError(
                f"npz heavy atom {k} has no coordinate match "
                f"(min dist={dist[j]:.4f}) in {npz_path}"
            )
        mask[j] = bool(scaffold_mask[k] == 1)
    return mask


def one_hot(x: List["str"], classes: List["str"]) -> torch.Tensor:
    return nn.functional.one_hot(
        torch.tensor([classes.index(s) for s in x], dtype=torch.long), len(classes)
    )


def atomic_symbols_to_one_hot(symbols: List[str]) -> torch.FloatTensor:
    return one_hot(symbols, atom_names)


def residue_names_to_one_hot(names: List[str]) -> torch.Tensor:
    return one_hot(names, residue_names)
