import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.pl_data import ProteinLigandData, torchify_dict  # noqa: E402
from utils.data import PDBProtein, parse_sdf_file               # noqa: E402


def load_complex(pdb_path, sdf_path, transform, pocket_radius=10):
    """Carve the pocket the way the training set was built (CrossDocked pocket10)."""
    ligand_dict = parse_sdf_file(sdf_path)
    protein = PDBProtein(pdb_path)
    n_all = len(protein.atoms)
    pocket_block = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand_dict, pocket_radius))
    pocket_dict = PDBProtein(pocket_block).to_dict_atom()
    print(f'pocket: {len(pocket_dict["element"])} atoms carved from {n_all} '
          f'(radius {pocket_radius} A)')
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict), ligand_dict=torchify_dict(ligand_dict))
    return transform(data), ligand_dict


@torch.no_grad()
def build_shift_table(model, net_cond, ref_pos, ref_v, protein_pos, protein_v,
                      batch_ligand, batch_protein, device):
    """h_bap frozen at the reference complex -> k_s[t] = k_t * s(t), shape [T, N, 3].

    BAPNet.extract_features centres by the pocket mean and consumes only pairwise
    distances, so its output is invariant to the rigid motion center_pos applies.
    """
    a_h = torch.argmax(protein_v[:, :6], dim=1)
    r_h = torch.argmax(protein_v[:, 6:26], dim=1)
    hbap_l, hbap_p = net_cond.extract_features(
        ref_pos, protein_pos, ref_v, a_h, r_h, batch_ligand, batch_protein)
    hbap_l, hbap_p = hbap_l.detach(), hbap_p.detach()

    T = model.num_timesteps
    s = torch.stack([
        model.shift_t_mlp_pos(
            torch.cat([hbap_l, torch.full((hbap_l.size(0), 1), float(t), device=device)], -1))
        for t in range(T)
    ], 0)                                                   # [T, N, 3]
    k_s = model.k_t.view(-1, 1, 1) * s
    return k_s.detach(), hbap_l, hbap_p


def to_real(z_tilde, k_s_t, gen_mask):
    """z~ -> z. Training only shifts the generated atoms (`ligand_pos_perturbed` keeps
    the retained scaffold at clean x0), so the shift must not touch the fixed part."""
    return z_tilde + torch.where(gen_mask.unsqueeze(-1), k_s_t, torch.zeros_like(k_s_t))
