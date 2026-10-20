import pdb
import os
import copy
import itertools
import logging
import re
import tempfile
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import BRICS
from rdkit import RDConfig
from rdkit import Geometry
from openbabel import openbabel as ob
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
try:
    import mdtraj
    import alphaspace2 as al
except:
    pass

logger = logging.getLogger('decompdiff_scaffold')
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s'))
    logger.addHandler(_handler)
logger.propagate = False

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}
RADIUS = 10
DECOMPDIFF_ALPHASPACE_CONFIG = {
    'beta_cluster_dist': 1.6,
    'contact_cutoff': 1.6,
    'pocket_cluster_dist': 6.0,
}
DECOMPDIFF_ALPHASPACE_SMALLER_POCKET_CONFIGS = [
    {
        'beta_cluster_dist': 1.6,
        'contact_cutoff': 1.4,
        'pocket_cluster_dist': 5.0,
    },
    {
        'beta_cluster_dist': 1.6,
        'contact_cutoff': 1.2,
        'pocket_cluster_dist': 4.0,
    },
    {
        'beta_cluster_dist': 1.6,
        'contact_cutoff': 1.0,
        'pocket_cluster_dist': 3.0,
    },
]
DECOMPDIFF_ALPHASPACE_LARGER_POCKET_CONFIGS = [
    {
        'beta_cluster_dist': 1.6,
        'contact_cutoff': 1.8,
        'pocket_cluster_dist': 7.0,
    },
    {
        'beta_cluster_dist': 1.6,
        'contact_cutoff': 2.0,
        'pocket_cluster_dist': 8.0,
    },
    {
        'beta_cluster_dist': 1.6,
        'contact_cutoff': 2.2,
        'pocket_cluster_dist': 9.0,
    },
]
DECOMPDIFF_GENERATED_MIN_RATIO = 0.3
DECOMPDIFF_GENERATED_MAX_RATIO = 0.8


def center_mol(mol, pre_center=None):
    conformer = mol.GetConformers()[0]
    pos = conformer.GetPositions()
    if pre_center is None: center = np.mean(pos, axis=0)
    else: center = pre_center
    conformer = mol.GetConformers()[0]
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conformer.GetAtomPosition(idx)

        p = Geometry.Point3D(float(pos.x - center[0]), float(pos.y - center[1]), float(pos.z - center[2]))
        conformer.SetAtomPosition(idx, p)
    return center

def load_ligand_mol(path, removeHs=True):
    mol = None
    last_exc = None
    for sanitize in (True, False):
        try:
            suppl = Chem.SDMolSupplier(path, removeHs=removeHs, sanitize=sanitize)
            mol = next(iter(suppl))
        except Exception as e:
            last_exc = e
            mol = None
        if mol is not None:
            return mol
    if last_exc is not None:
        raise last_exc
    raise ValueError(f'Failed to read ligand molecule from {path}')

def parse_sdf_file(path, center_ligand=False, pre_center=None):
    rdmol = load_ligand_mol(path, removeHs=True)
    if center_ligand:
        center = center_mol(rdmol, pre_center=pre_center)
    else:
        center = None
    return parse_rdkit_mol(rdmol, center=center), rdmol


def convert_sdf_to_pdb(sdf_path, pdb_path):
    ob_conversion = ob.OBConversion()
    ob_conversion.SetInAndOutFormats("sdf", "pdb")
    mol = ob.OBMol()
    ob_conversion.ReadFile(mol, sdf_path)
    ob_conversion.WriteFile(mol, pdb_path)


def decompose_molecule(mol, method='BRICS'):
    mol = copy.deepcopy(mol)
    if method != 'BRICS':
        raise NotImplementedError

    raw_frags_smiles = BRICS.BRICSDecompose(mol)
    frags_smiles, frags_atom_idx = [], []
    for smiles in list(raw_frags_smiles):
        rogue_smiles = re.sub(r'\[[0-9]+\*\]', '[H]', smiles)
        rogue_smiles = re.sub(r'\(\)', '', rogue_smiles)
        rogue_frag = Chem.MolFromSmiles(rogue_smiles)
        if rogue_frag is None or not mol.HasSubstructMatch(rogue_frag):
            continue
        frags_smiles.append(rogue_smiles)
        frags_atom_idx.append(mol.GetSubstructMatches(rogue_frag))

    if len(frags_smiles) == 0:
        raise ValueError('BRICS decomposition failed to produce fragments.')

    sorted_result = sorted(zip(frags_smiles, frags_atom_idx), key=lambda x: len(x[1]))
    frags_smiles, frags_atom_idx = zip(*sorted_result)
    return frags_smiles, frags_atom_idx


def find_complete_seg(current_idx_set, current_match_list, all_atom_idx, num_element):
    if len(all_atom_idx) == 0:
        if len(current_idx_set) == num_element:
            return current_match_list
        return None

    raw_matches = all_atom_idx[0]
    all_matches_subset = []
    matches = []
    for match in raw_matches:
        if any(x in current_idx_set for x in match):
            continue
        matches.append(match)

    for size in reversed(range(1, min(len(matches) + 1, num_element - len(current_idx_set) + 1))):
        for subset in itertools.combinations(matches, size):
            subset = list(itertools.chain(*subset))
            if len(subset) == len(set(subset)) and \
                    len(set(subset + list(current_idx_set))) == len(subset) + len(current_idx_set):
                all_matches_subset.append(subset)

    for match in all_matches_subset:
        next_idx_set = copy.deepcopy(current_idx_set)
        next_match_list = copy.deepcopy(current_match_list)
        for atom_idx in match:
            next_idx_set.add(atom_idx)
        next_match_list.append(match)
        match_list = find_complete_seg(next_idx_set, next_match_list, all_atom_idx[1:], num_element)
        if match_list is not None:
            return match_list


def compute_pocket_frag_distance(pocket_centers, frag_centroid):
    return np.mean([np.linalg.norm(frag_centroid - center, ord=2) for center in pocket_centers])


def is_terminal_frag(mol, frag_atom_idx):
    split_bond_idx = []
    for bond_idx, bond in enumerate(mol.GetBonds()):
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if (start in frag_atom_idx) != (end in frag_atom_idx):
            split_bond_idx.append(bond_idx)
    return len(split_bond_idx) <= 1


def get_submol(mol, split_bond_idx, pocket_atom_idx):
    if len(pocket_atom_idx) == 0:
        return None
    if len(pocket_atom_idx) == mol.GetNumAtoms() and len(split_bond_idx) == 0:
        return copy.deepcopy(mol)
    r = Chem.FragmentOnBonds(mol, split_bond_idx)
    frags = Chem.GetMolFrags(r)
    frags_overlap_atoms = [len(set(pocket_atom_idx).intersection(set(frag))) for frag in frags]
    hit_idx = np.argmax(frags_overlap_atoms)
    return Chem.GetMolFrags(r, asMols=True)[hit_idx]


def extract_submols(mol, pocket_list, ligand_path=None, label='attempt'):
    union_frags_smiles, possible_frags_atom_idx = decompose_molecule(mol)
    match_frags_list = find_complete_seg(set(), [], possible_frags_atom_idx, mol.GetNumAtoms())
    if match_frags_list is None:
        raise ValueError('Failed to find a complete fragment covering.')

    frags_smiles_list, frags_atom_idx_list = [], []
    for smiles, group_atom_idx in zip(union_frags_smiles, match_frags_list):
        query_frag_mol = Chem.MolFromSmiles(smiles)
        if query_frag_mol is None:
            continue
        if len(group_atom_idx) == query_frag_mol.GetNumAtoms():
            frags_smiles_list.append(smiles)
            frags_atom_idx_list.append(group_atom_idx)
        else:
            assert len(group_atom_idx) % query_frag_mol.GetNumAtoms() == 0
            n_atoms = 0
            for match in mol.GetSubstructMatches(query_frag_mol):
                if all(atom_idx in group_atom_idx for atom_idx in match):
                    frags_smiles_list.append(smiles)
                    frags_atom_idx_list.append([atom_idx for atom_idx in match])
                    n_atoms += len(match)
            assert n_atoms == len(group_atom_idx)

    ligand_pos = mol.GetConformer().GetPositions()
    dist_mat = np.zeros([len(frags_smiles_list), len(pocket_list)])
    all_frag_centroid = []
    for frag_idx, frag_atom_idx in enumerate(frags_atom_idx_list):
        frag_pos = np.array([ligand_pos[atom_idx] for atom_idx in frag_atom_idx])
        frag_centroid = np.mean(frag_pos, 0)
        all_frag_centroid.append(frag_centroid)
        for pocket_idx, pocket in enumerate(pocket_list):
            centers = [a.centroid for a in pocket.alphas]
            dist_mat[frag_idx, pocket_idx] = compute_pocket_frag_distance(centers, frag_centroid)
    all_frag_centroid = np.array(all_frag_centroid)

    terminal_mask = np.array([is_terminal_frag(mol, v) for v in frags_atom_idx_list])
    t_frag_idx = (terminal_mask == 1).nonzero()[0]
    pocket_idx, frag_idx = linear_sum_assignment(dist_mat[t_frag_idx].T)
    arms_frag_idx = np.array([t_frag_idx[idx] for idx in frag_idx])
    clustering_centers = [all_frag_centroid[idx] for idx in arms_frag_idx]
    cluster_pocket_idx = list(pocket_idx)
    if len(clustering_centers) < len(pocket_list):
        add_pocket_idx = list(set(range(len(pocket_list))) - set(pocket_idx))
        for p_idx in add_pocket_idx:
            centers = [a.centroid for a in pocket_list[p_idx].alphas]
            clustering_centers.append(np.mean(centers, 0))
            cluster_pocket_idx.append(p_idx)

    non_arm_frag_idx = np.array([idx for idx in range(len(all_frag_centroid)) if idx not in arms_frag_idx])
    if len(non_arm_frag_idx) > 0:
        scaffold_frag_idx = non_arm_frag_idx[
            np.argmax(distance_matrix(all_frag_centroid[non_arm_frag_idx], clustering_centers).sum(-1))
        ]
        clustering_centers.append(all_frag_centroid[scaffold_frag_idx])
    else:
        scaffold_frag_idx = np.array([], dtype=int)

    frag_cluster_dist_mat = distance_matrix(all_frag_centroid, clustering_centers)
    assignment = -1 * np.ones(len(all_frag_centroid)).astype(np.int64)
    assignment[arms_frag_idx] = pocket_idx
    assignment[scaffold_frag_idx] = len(clustering_centers) - 1

    for idx in range(len(all_frag_centroid)):
        assign_cluster_idx = frag_cluster_dist_mat[idx].argmin()
        if assign_cluster_idx == len(clustering_centers) - 1:
            assignment[idx] = len(clustering_centers) - 1
        else:
            assign_pocket_idx = cluster_pocket_idx[assign_cluster_idx]
            current_atom_idx = []
            for assign_frag_idx in (assignment == assign_pocket_idx).nonzero()[0]:
                current_atom_idx += frags_atom_idx_list[assign_frag_idx]
            current_atom_idx += frags_atom_idx_list[idx]
            if is_terminal_frag(mol, current_atom_idx):
                assignment[idx] = assign_pocket_idx
            else:
                assignment[idx] = len(clustering_centers) - 1

    scaffold_atom_idx = []
    for assigned_idx in (assignment == len(pocket_list)).nonzero()[0]:
        scaffold_atom_idx += frags_atom_idx_list[assigned_idx]
    return scaffold_atom_idx


def is_valid_decompdiff_generated_ratio(scaffold_mask):
    num_total = int(scaffold_mask.size)
    if num_total <= 0:
        return False
    ratio = get_decompdiff_generated_ratio(scaffold_mask)
    return DECOMPDIFF_GENERATED_MIN_RATIO <= ratio <= DECOMPDIFF_GENERATED_MAX_RATIO


def get_decompdiff_generated_ratio(scaffold_mask):
    num_total = int(scaffold_mask.size)
    if num_total <= 0:
        return 0.0
    return float(scaffold_mask.sum()) / num_total


def get_decompdiff_generated_ratio_distance(scaffold_mask):
    ratio = get_decompdiff_generated_ratio(scaffold_mask)
    if ratio < DECOMPDIFF_GENERATED_MIN_RATIO:
        return DECOMPDIFF_GENERATED_MIN_RATIO - ratio
    if ratio > DECOMPDIFF_GENERATED_MAX_RATIO:
        return ratio - DECOMPDIFF_GENERATED_MAX_RATIO
    return 0.0


def get_decompdiff_generated_ratio_status(scaffold_mask):
    num_total = int(scaffold_mask.size)
    if num_total <= 0:
        return 'invalid'
    ratio = get_decompdiff_generated_ratio(scaffold_mask)
    if ratio < DECOMPDIFF_GENERATED_MIN_RATIO:
        return 'too_small'
    if ratio > DECOMPDIFF_GENERATED_MAX_RATIO:
        return 'too_large'
    return 'valid'


def format_alphaspace_config(alphaspace_config):
    return ', '.join(f'{key}={value}' for key, value in sorted(alphaspace_config.items()))


def log_decompdiff_scaffold_ratio_attempts(attempted_masks):
    ratio_items = [
        f'{get_decompdiff_generated_ratio(mask):.4f}'
        for mask in attempted_masks
    ]
    logger.info(f'ratios: [{", ".join(ratio_items)}]')


def log_selected_decompdiff_scaffold_ratio(scaffold_mask):
    logger.info(f'selected_ratio: {get_decompdiff_generated_ratio(scaffold_mask):.4f}')


def compute_decompdiff_scaffold_mask_with_config(receptor, binder, mol, ligand_path, alphaspace_config, label='attempt'):
    ss = al.Snapshot()
    for key, value in alphaspace_config.items():
        setattr(ss, key, value)
    ss.run(receptor=receptor, binder=binder)
    pocket_list = [p for p in sorted(ss.pockets, key=lambda i: i.nonpolar_space, reverse=True) if p.isContact]
    if len(pocket_list) == 0:
        raise RuntimeError(f'AlphaSpace did not find any contact pocket for {ligand_path}.')

    scaffold_atom_idx = extract_submols(
        mol,
        pocket_list,
        ligand_path=ligand_path,
        label=label,
    )
    scaffold_mask = np.zeros(mol.GetNumAtoms(), dtype=np.int32)
    scaffold_mask[np.asarray(scaffold_atom_idx, dtype=np.int64)] = 1
    return scaffold_mask


def compute_decompdiff_scaffold_mask(protein_path, ligand_path, rdmol):
    tmp_pdb_path = None
    try:
        mol = load_ligand_mol(ligand_path, removeHs=True)
        if mol is None:
            mol = Chem.Mol(rdmol)
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
            tmp_pdb_path = tmp.name
        convert_sdf_to_pdb(ligand_path, tmp_pdb_path)

        receptor = mdtraj.load(protein_path)
        binder = mdtraj.load(tmp_pdb_path)
        scaffold_mask = compute_decompdiff_scaffold_mask_with_config(
            receptor=receptor,
            binder=binder,
            mol=mol,
            ligand_path=ligand_path,
            alphaspace_config=DECOMPDIFF_ALPHASPACE_CONFIG,
            label='initial',
        )
        attempted_masks = [scaffold_mask]
        attempted_labels = ['initial']
        status = get_decompdiff_generated_ratio_status(scaffold_mask)
        if status == 'valid':
            log_decompdiff_scaffold_ratio_attempts(attempted_masks)
            log_selected_decompdiff_scaffold_ratio(scaffold_mask)
            return scaffold_mask

        if status == 'too_small':
            retry_groups = [
                ('too_small_larger_pocket', DECOMPDIFF_ALPHASPACE_LARGER_POCKET_CONFIGS),
            ]
        elif status == 'too_large':
            retry_groups = [
                ('too_large_smaller_pocket', DECOMPDIFF_ALPHASPACE_SMALLER_POCKET_CONFIGS),
            ]
        else:
            retry_groups = []

        for retry_group, retry_configs in retry_groups:
            for retry_idx, alphaspace_config in enumerate(retry_configs, start=1):
                label = f'{retry_group}_retry_{retry_idx}'
                retry_mask = compute_decompdiff_scaffold_mask_with_config(
                    receptor=receptor,
                    binder=binder,
                    mol=mol,
                    ligand_path=ligand_path,
                    alphaspace_config=alphaspace_config,
                    label=label,
                )
                attempted_masks.append(retry_mask)
                attempted_labels.append(label)
                if is_valid_decompdiff_generated_ratio(retry_mask):
                    log_decompdiff_scaffold_ratio_attempts(attempted_masks)
                    log_selected_decompdiff_scaffold_ratio(retry_mask)
                    return retry_mask

        best_idx, best_mask = min(
            enumerate(attempted_masks),
            key=lambda item: (get_decompdiff_generated_ratio_distance(item[1]), item[0])
        )
        log_decompdiff_scaffold_ratio_attempts(attempted_masks)
        log_selected_decompdiff_scaffold_ratio(best_mask)
        return best_mask
    except Exception as e:
        raise RuntimeError(
            f'DecompDiff-style scaffold extraction failed for ligand {ligand_path}: {type(e).__name__}: {e}'
        ) from e
    finally:
        if tmp_pdb_path is not None and os.path.exists(tmp_pdb_path):
            os.remove(tmp_pdb_path)

# def parse_rdkit_mol(rdmol, center=None):
#     fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
#     factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
#     # Remove Hydrogens.
#     rd_num_atoms = rdmol.GetNumAtoms()
#     feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
#     for feat in factory.GetFeaturesForMol(rdmol):
#         feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

#     # Get hybridization in the order of atom idx.
#     hybridization = []
#     for atom in rdmol.GetAtoms():
#         hybr = str(atom.GetHybridization())
#         idx = atom.GetIdx()
#         hybridization.append((idx, hybr))
#     hybridization = sorted(hybridization)
#     hybridization = [v[1] for v in hybridization]

#     ptable = Chem.GetPeriodicTable()

#     pos = np.array(rdmol.GetConformers()[-1].GetPositions(), dtype=np.float32)
#     element = []
#     accum_pos = 0
#     accum_mass = 0
#     for atom_idx in range(rd_num_atoms):
#         atom = rdmol.GetAtomWithIdx(atom_idx)
#         atom_num = atom.GetAtomicNum()
#         element.append(atom_num)
#         atom_weight = ptable.GetAtomicWeight(atom_num)
#         accum_pos += pos[atom_idx] * atom_weight
#         accum_mass += atom_weight
#     center_of_mass = accum_pos / accum_mass
#     element = np.array(element, dtype=np.int32)

#     # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
#     row, col, edge_type = [], [], []
#     for bond in rdmol.GetBonds():
#         start = bond.GetBeginAtomIdx()
#         end = bond.GetEndAtomIdx()
#         row += [start, end]
#         col += [end, start]
#         edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

#     edge_index = np.array([row, col], dtype=np.int64)
#     edge_type = np.array(edge_type, dtype=np.int64)

#     perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
#     edge_index = edge_index[:, perm]
#     edge_type = edge_type[perm]

#     data = {
#         'smiles': Chem.MolToSmiles(rdmol),
#         'element': element,
#         'pos': pos,
#         'bond_index': edge_index,
#         'bond_type': edge_type,
#         'center_of_mass': center_of_mass,
#         'atom_feature': feat_mat,
#         'hybridization': hybridization,
#         'center': center
#          #'shape_emb': shape_emb,
#     }
#     return data


def parse_rdkit_mol(rdmol, center=None):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    # Remove Hydrogens (if needed)
    rd_num_atoms = rdmol.GetNumAtoms()

    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    # Hybridization
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybridization.append((atom.GetIdx(), str(atom.GetHybridization())))
    hybridization = [v[1] for v in sorted(hybridization)]

    # Positions
    pos = np.array(rdmol.GetConformers()[-1].GetPositions(), dtype=np.float32)

    # Element + center of mass
    ptable = Chem.GetPeriodicTable()
    element = []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight

    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int32)

    # Bonds
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.int64)
    edge_type = np.array(edge_type, dtype=np.int64)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    scaffold_mask = np.zeros(rd_num_atoms, dtype=np.int32)

    # try:
    #     scaffold = MurckoScaffold.GetScaffoldForMol(rdmol)
    #     if scaffold is not None and scaffold.GetNumAtoms() > 0:
    #         match = rdmol.GetSubstructMatch(scaffold)
    #         for idx in match:
    #             scaffold_mask[idx] = 1
    # except Exception:
    #     pass

    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization,
        'center': center,
        'scaffold_mask': scaffold_mask
    }

    return data


def augment_scaffold_mask_by_pocket_distance(ligand_dict, pocket_residues, protein, dist_thresh=3.0):
    scaffold_mask = np.array(ligand_dict['scaffold_mask'], dtype=np.int32, copy=True)
    if scaffold_mask.size == 0 or len(pocket_residues) == 0:
        return scaffold_mask

    ligand_pos = np.asarray(ligand_dict['pos'], dtype=np.float32)
    if ligand_pos.size == 0:
        return scaffold_mask
    ligand_center = ligand_dict.get('center', None)
    if ligand_center is not None:
        ligand_pos = ligand_pos + np.asarray(ligand_center, dtype=np.float32)

    pocket_residue_com = []
    for residue in pocket_residues:
        if 'center_of_mass' in residue:
            pocket_residue_com.append(residue['center_of_mass'])

    if len(pocket_residue_com) == 0:
        return scaffold_mask

    pocket_residue_com = np.asarray(pocket_residue_com, dtype=np.float32)
    diff = ligand_pos[:, None, :] - pocket_residue_com[None, :, :]
    min_dist = np.linalg.norm(diff, axis=-1).min(axis=1)

    scaffold_mask[min_dist < dist_thresh] = 0
    return scaffold_mask

def compute_scaffold_mask_from_reference(ligand_dict, ref_ligand_path, tol=1e-3):
    """Derive a scaffold mask by position-matching against a reference ligand.

    The overlapping atoms (identical 3D positions in both the current ligand and
    the reference ligand) are treated as the FIXED part (mask value 0), and the
    remaining atoms as the GENERATED part (mask value 1), matching the
    ``generated_mask = scaffold_mask == 1`` convention used downstream.

    Matching is done in absolute SDF coordinates so that the two molecules'
    differing per-molecule centers do not affect the result.
    """
    cur_abs = np.asarray(ligand_dict['pos'], dtype=np.float64)
    if ligand_dict.get('center', None) is not None:
        cur_abs = cur_abs + np.asarray(ligand_dict['center'], dtype=np.float64)

    ref_dict, _ = parse_sdf_file(ref_ligand_path, center_ligand=False)
    ref_abs = np.asarray(ref_dict['pos'], dtype=np.float64)
    if ref_abs.size == 0:
        raise RuntimeError(f'reference ligand has no coordinates: {ref_ligand_path}')

    # [N, M] pairwise distances between current atoms and reference atoms.
    dists = np.linalg.norm(cur_abs[:, None, :] - ref_abs[None, :, :], axis=-1)
    matched = dists.min(axis=1) < float(tol)

    scaffold_mask = np.where(matched, 0, 1).astype(np.int32)
    num_fixed = int(matched.sum())
    if num_fixed == 0:
        raise RuntimeError(
            f'no atoms of {ligand_dict.get("filename", "ligand")} matched reference '
            f'{ref_ligand_path} within tol={tol}; cannot derive fixed part.'
        )
    if num_fixed == scaffold_mask.size:
        raise RuntimeError(
            f'all atoms matched reference {ref_ligand_path} within tol={tol}; '
            f'generated part would be empty.'
        )
    print(
        f'[ref_scaffold_mask] matched_fixed={num_fixed}, '
        f'generated={scaffold_mask.size - num_fixed}, total={scaffold_mask.size}, '
        f'ref={ref_ligand_path}, tol={tol}'
    )
    return scaffold_mask


def parse_pdb(protein_path, ligand_path, ref_ligand_path=None, ref_match_tol=1e-3):
    protein = PDBProtein(protein_path)
    if ligand_path == None:
        pocket_dict = protein.residues_to_atom_dict(protein.residues)
    else:
        ligand_dict, rdmol = parse_sdf_file(ligand_path, center_ligand=True)
        if ref_ligand_path is not None:
            # Reference-driven scaffold mask: failures should be loud, not silently
            # swallowed, since the caller explicitly asked for this mode.
            decomp_scaffold_mask = compute_scaffold_mask_from_reference(
                ligand_dict, ref_ligand_path, tol=ref_match_tol
            )
            if len(decomp_scaffold_mask) != len(ligand_dict['element']):
                raise RuntimeError(
                    f'scaffold_mask size mismatch: got {len(decomp_scaffold_mask)}, expected {len(ligand_dict["element"])}'
                )
            ligand_dict['scaffold_mask'] = decomp_scaffold_mask
        else:
            try:
                decomp_scaffold_mask = compute_decompdiff_scaffold_mask(protein_path, ligand_path, rdmol)
                if len(decomp_scaffold_mask) != len(ligand_dict['element']):
                    raise RuntimeError(
                        f'scaffold_mask size mismatch: got {len(decomp_scaffold_mask)}, expected {len(ligand_dict["element"])}'
                    )
                ligand_dict['scaffold_mask'] = decomp_scaffold_mask
            except Exception as e:
                print(
                    f'Warning: failed to compute DecompDiff scaffold mask for {ligand_path}: '
                    f'{type(e).__name__}: {e}'
                )
        # pocket_dict = protein.residues_to_atom_dict(protein.residues)
        pocket_residues = protein.query_residues_ligand(ligand_dict['pos'] + ligand_dict['center'])
        # ligand_dict['scaffold_mask'] = augment_scaffold_mask_by_pocket_distance(
        #     ligand_dict, pocket_residues, protein, dist_thresh=4.0
        # )
        pocket_dict = protein.residues_to_atom_dict(pocket_residues)
    return pocket_dict, ligand_dict


def save_residues_around_reference_ligand(protein_path, ligand_path, out_path, radius=10.0,
                                          criterion='center_of_mass', name='POCKET'):
    protein = PDBProtein(protein_path)
    ligand_dict, _ = parse_sdf_file(ligand_path, center_ligand=False)
    ligand_pos = np.asarray(ligand_dict['pos'], dtype=np.float32)
    if ligand_pos.size == 0:
        raise ValueError(f'Ligand has no coordinates: {ligand_path}')

    pocket_residues = protein.query_residues_ligand(
        ligand_pos=ligand_pos,
        radius=radius,
        criterion=criterion,
    )
    pdb_block = protein.residues_to_pdb_block(pocket_residues, name=name)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(pdb_block)
    return pocket_residues

def parse_sdf_mol_with_pdb(index, raw_path, center_ligand=True):
    protein_fn, ligand_fn = index[:2]
    tmp = "/"+"/".join(ligand_fn.split("/")[-2:])
    ligand_dict, rdmol = parse_sdf_file(raw_path+tmp, center_ligand=center_ligand)
    tmp = "/"+"/".join(protein_fn.split("/")[-2:])
    protein_path = raw_path + tmp
    try:
        decomp_scaffold_mask = compute_decompdiff_scaffold_mask(protein_path, raw_path + "/" + "/".join(ligand_fn.split("/")[-2:]), rdmol)
        if len(decomp_scaffold_mask) != len(ligand_dict['element']):
            raise RuntimeError(
                f'scaffold_mask size mismatch: got {len(decomp_scaffold_mask)}, expected {len(ligand_dict["element"])}'
            )
        ligand_dict['scaffold_mask'] = decomp_scaffold_mask
    except Exception as e:
        print(
            f'Warning: failed to compute DecompDiff scaffold mask for {raw_path + "/" + "/".join(ligand_fn.split("/")[-2:])}: '
            f'{type(e).__name__}: {e}'
        )
    protein = PDBProtein(protein_path)
    if center_ligand:
        pocket_residues = protein.query_residues_ligand(ligand_dict['pos'] + ligand_dict['center'])
    else:
        pocket_residues = protein.query_residues_ligand(ligand_dict['pos'])
    # ligand_dict['scaffold_mask'] = augment_scaffold_mask_by_pocket_distance(
    #     ligand_dict, pocket_residues, protein, dist_thresh=4.0
    # )
    pocket_dict = protein.residues_to_atom_dict(pocket_residues)
    """
    try:
        protein_fn, ligand_fn = index[:2]
        tmp = "/"+"/".join(ligand_fn.split("/")[-2:])
        ligand_dict, rdmol = parse_sdf_file(raw_path+tmp, center_ligand=center_ligand)
        tmp = "/"+"/".join(protein_fn.split("/")[-2:])
        protein = PDBProtein(raw_path+tmp)
        if center_ligand:
            pocket_dict = protein.residues_to_atom_dict(
                protein.query_residues_ligand(ligand_dict['pos']+ligand_dict['center'])
            )
        else:
            pocket_dict = protein.residues_to_atom_dict(
                protein.query_residues_ligand(ligand_dict['pos'])
            )
    except Exception as e:
        print(e)
        protein_fn, ligand_fn, ligand_dict, pocket_dict, rdmol = None, None, None, None, None
    """
    return (protein_fn, ligand_fn, ligand_dict, pocket_dict, rdmol)

class PDBProtein(object):
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass

        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int64),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand_pos, radius=RADIUS, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand_pos:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block
    
    def residues_to_atom_dict(self, residues, name='POCKET'):
        element, pos, is_backbone, atom_name, atom_to_aa_type = [], [], [], [], []
        for residue in residues:
            for atom_idx in residue['atoms']:
                atom = self.atoms[atom_idx]
                atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
                element.append(atomic_number)
                pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
                atom_name.append(atom['atom_name'])
                is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
                atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

        return {
            'element': np.array(element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(pos, dtype=np.float32),
            'is_backbone': np.array(is_backbone, dtype=np.bool_),
            'atom_name': atom_name,
            'atom_to_aa_type': np.array(atom_to_aa_type, dtype=np.int64)
        }




# save_residues_around_reference_ligand(
#     protein_path='/fs/ess/PCON0041/gruoxi/SBDDcode/test_data/xxx2/xxx2_protein.pdb',
#     ligand_path='/fs/ess/PCON0041/gruoxi/SBDDcode/test_data/xxx2/xxx2_ligand.sdf',
#     out_path='/fs/ess/PCON0041/gruoxi/SBDDcode/test_data/xxx2/xxx2_pocket.pdb',
#     radius=10.0,
# )
