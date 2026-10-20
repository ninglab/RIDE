import os
import sys
from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analysis.build import MoleculeBuilder                    # noqa: E402
from data import Ligand, Protein, ProteinLigandComplex        # noqa: E402
from data.featurization import ProteinLigandSimpleFeaturization  # noqa: E402
from data.transform import ObabelTransform, ReduceTransform   # noqa: E402
from model import DiffusionHoppingModel                       # noqa: E402


# ---------------------------------------------------------------------------
# CoM-free subspace helpers
# ---------------------------------------------------------------------------


def com_project(v):
    """Project ``[..., n, 3]`` onto the zero-centre-of-mass subspace."""
    return v - v.mean(dim=-2, keepdim=True)


def com_free_randn(shape, device=None, dtype=torch.float32, generator=None):
    """Standard normal projected onto the CoM-free subspace.

    Deliberately NOT rescaled to restore unit per-coordinate variance: DiffHopp
    itself injects ``centered_batch(randn_like(...))`` in ``q``/``p``, so the
    unscaled projection is the convention the checkpoint was trained with.
    """
    z = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return com_project(z)


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------


def load_model(checkpoint_path, device):
    model = DiffusionHoppingModel.load_from_checkpoint(
        checkpoint_path, map_location=device
    ).to(device)
    model.eval()
    model.freeze()
    return model


def load_complex(pdb_path, sdf_path, scaffold_npz=None):
    """Featurize one protein-ligand complex, optionally with an SBDDcode npz mask.

    Mirrors ``generate_scaffolds.py``: SDF ligands are used as-is (an sdf->sdf
    openbabel round trip can perturb the molecule and break the coordinate match
    against the scaffold npz), everything else goes through openbabel.
    """
    pdb_path = Path(pdb_path)
    sdf_path = Path(sdf_path)
    if sdf_path.suffix == ".sdf":
        ligand_path = sdf_path
    else:
        ligand_path = ObabelTransform(from_format=sdf_path.suffix[1:], to_format="sdf")(sdf_path)

    protein_transform = Compose([ObabelTransform(), ReduceTransform()])
    protein = Protein(protein_transform(pdb_path))
    ligand = Ligand(ligand_path)

    if scaffold_npz is not None:
        npz_path = str(scaffold_npz)
        identifier = os.path.splitext(os.path.basename(npz_path))[0]
        scaffold_npz_dir = os.path.dirname(npz_path)
    else:
        identifier = "complex"
        scaffold_npz_dir = None

    pl_complex = ProteinLigandComplex(protein, ligand, identifier=identifier)
    featurization = ProteinLigandSimpleFeaturization(
        c_alpha_only=True, cutoff=8.0, mode="residue", scaffold_npz_dir=scaffold_npz_dir
    )
    return featurization(pl_complex), ligand


# ---------------------------------------------------------------------------
# Diffusion primitives
# ---------------------------------------------------------------------------


class ScaffoldDiffusion:
    """Batched reverse/forward primitives for one protein-ligand complex.

    Wraps ``DiffusionHoppingModel.model`` without modifying the library: the
    reverse mean and standard deviation that ``DiffusionModel.p`` computes
    inline are re-derived here so that noise can be inferred and re-injected.
    """

    def __init__(self, lit_model, data, device, dtype=torch.float32):
        # float64 is what makes the round trip survive a 500-step chain. The reverse
        # step is exact by construction; it drifts only because two identical forward
        # passes disagree at ~1 ULP, and that seed is amplified geometrically. In
        # float32 the seed is ~1e-6 A; in float64 it is ~1e-15. DiffHopp has no
        # categorical channel, so unlike SBDDcode/IPDiff it has no argmax quantisation
        # to absorb sub-threshold drift -- every ULP just accumulates.
        self.dtype = dtype
        self.lit_model = lit_model
        self.dm = lit_model.model
        if dtype == torch.float64:
            self.lit_model = self.lit_model.double()
            self.dm = self.lit_model.model
            bad = [n for n, q in self.dm.named_parameters() if q.dtype != torch.float64]
            if bad:
                raise RuntimeError(
                    f'{len(bad)} parameters stayed float32 after .double(), e.g. '
                    f'{bad[:5]}. They are not reachable from the module tree.')
        self.schedule = self.dm.schedule
        self.T = int(self.dm.T)
        self.device = device

        self.pos_norm = float(self.dm.pos_norm)
        self.x_norm = float(self.dm.x_norm)
        self.x_bias = float(self.dm.x_bias)
        if not self.dm.condition_on_fg:
            raise ValueError(
                "checkpoint was trained with condition_on_fg=False; scaffold masking "
                "is not applied by the model and this engine would be meaningless."
            )

        self.data = data
        self.gen_mask = data["ligand"].scaffold_mask.view(-1).bool().to(device)
        self.num_atoms = int(self.gen_mask.numel())
        self.num_gen = int(self.gen_mask.sum().item())
        self.num_features = int(data["ligand"].x.shape[-1])

        # Frame: subtract the generated-atom CoM of x_0, then normalize.
        pos_lig = data["ligand"].pos.to(device).to(self.dtype)
        self.mean = pos_lig[self.gen_mask].mean(dim=0)
        self.x0_pos_frame = ((pos_lig - self.mean) / self.pos_norm).to(device)
        self.x0_x_frame = (data["ligand"].x.to(device).to(self.dtype) / self.x_norm
                           + self.x_bias)

        self._batch_cache = {}

    @property
    def pos_dof(self):
        """Dimension of the CoM-free position subspace of the generated atoms."""
        return 3 * (self.num_gen - 1)

    @property
    def x_dof(self):
        return self.num_features * self.num_gen

    def get_batch(self, batch_size):
        """``batch_size`` copies of the complex, already moved into the frame."""
        batch_size = int(batch_size)
        if batch_size in self._batch_cache:
            return self._batch_cache[batch_size]
        batch = Batch.from_data_list([self.data] * batch_size).to(self.device)
        batch["ligand"].pos = (
            (batch["ligand"].pos - self.mean) / self.pos_norm
        ).contiguous()
        batch["protein"].pos = (
            (batch["protein"].pos - self.mean) / self.pos_norm
        ).contiguous()
        batch["ligand"].x = (
            batch["ligand"].x.to(self.dtype) / self.x_norm + self.x_bias
        ).contiguous()
        batch["protein"].x = (
            batch["protein"].x.to(self.dtype) / self.x_norm + self.x_bias
        ).contiguous()
        self._batch_cache[batch_size] = batch
        return batch

    def flat_mask(self, batch_size):
        return (
            self.gen_mask.unsqueeze(0)
            .expand(int(batch_size), self.num_atoms)
            .reshape(-1)
            .contiguous()
        )

    def initial_state(self, batch_size):
        """``x_0`` replicated ``batch_size`` times, in frame coordinates."""
        pos = self.x0_pos_frame.unsqueeze(0).repeat(int(batch_size), 1, 1).clone()
        x = self.x0_x_frame.unsqueeze(0).repeat(int(batch_size), 1, 1).clone()
        return pos, x

    @torch.no_grad()
    def eps_hat(self, pos, x, t_scalar, batch=None):
        """Network noise estimate on the generated atoms.

        Returns ``(eps_x, eps_pos)`` shaped ``[B, N_gen, .]``. The position
        estimate is projected onto the CoM-free subspace: the estimator
        returns its coordinate output uncentred, and only ``DiffusionModel.p``
        re-imposes the constraint downstream.
        """
        batch_size = int(pos.shape[0])
        if batch is None:
            batch = self.get_batch(batch_size)
        batch["ligand"].pos = pos.reshape(-1, 3).to(self.device)
        batch["ligand"].x = x.reshape(-1, self.num_features).to(self.device)

        mask = self.flat_mask(batch_size)
        t = torch.full(
            (batch["ligand"].num_nodes, 1),
            int(t_scalar),
            device=self.device,
            dtype=torch.long,
        )
        eps_x, eps_pos = self.dm.estimator(batch, t / self.T, mask)
        eps_x = eps_x.view(batch_size, self.num_gen, self.num_features)
        eps_pos = com_project(eps_pos.view(batch_size, self.num_gen, 3))
        return eps_x, eps_pos

    @torch.no_grad()
    def p_mean_std(self, pos, x, t_scalar, batch=None):
        """Posterior mean/std of ``p(z_{t-1} | z_t)`` on the generated atoms.

        Reproduces ``DiffusionModel.p`` for ``Parametrization.EPS`` exactly,
        including the CoM projection of the position mean. ``sigma`` is a
        scalar for a scalar ``t``.
        """
        t_scalar = int(t_scalar)
        eps_x, eps_pos = self.eps_hat(pos, x, t_scalar, batch=batch)

        beta_t = self.schedule.beta[t_scalar]
        sqrt_one_minus_alpha_bar_t = self.schedule.sqrt_one_minus_alpha_bar[t_scalar]
        sqrt_recip_alpha_t = self.schedule.sqrt_recip_alpha[t_scalar]

        gen = self.gen_mask
        mu_x = sqrt_recip_alpha_t * (
            x[:, gen] - beta_t * eps_x / sqrt_one_minus_alpha_bar_t
        )
        mu_pos = sqrt_recip_alpha_t * (
            pos[:, gen] - beta_t * eps_pos / sqrt_one_minus_alpha_bar_t
        )
        mu_pos = com_project(mu_pos)
        sigma = self.schedule.posterior_variance[t_scalar].sqrt()
        return mu_x, mu_pos, sigma

    @torch.no_grad()
    def ddim_forward_step(self, pos, x, t_scalar, t_next, batch=None):
        """One deterministic DDIM step from index ``t_scalar`` to ``t_next``.

        ``eps_pos`` is CoM-projected inside ``eps_hat``, which keeps both
        ``z0_hat`` and the produced state inside the CoM-free subspace --
        without that projection the drift compounds over the full chain.
        """
        eps_x, eps_pos = self.eps_hat(pos, x, t_scalar, batch=batch)
        sqrt_ab_t = self.schedule.sqrt_alpha_bar[int(t_scalar)]
        sqrt_1mab_t = self.schedule.sqrt_one_minus_alpha_bar[int(t_scalar)]
        sqrt_ab_n = self.schedule.sqrt_alpha_bar[int(t_next)]
        sqrt_1mab_n = self.schedule.sqrt_one_minus_alpha_bar[int(t_next)]

        gen = self.gen_mask
        x0_x = (x[:, gen] - sqrt_1mab_t * eps_x) / sqrt_ab_t
        x0_pos = (pos[:, gen] - sqrt_1mab_t * eps_pos) / sqrt_ab_t

        next_pos = pos.clone()
        next_x = x.clone()
        next_x[:, gen] = sqrt_ab_n * x0_x + sqrt_1mab_n * eps_x
        # Re-project rather than trusting the algebra to preserve the subspace:
        # 1/sqrt(alpha_bar_t) grows to ~100 at the noisy end of a full 500-step chain,
        # so float32 residue in the centre of mass is amplified as it accumulates.
        next_pos[:, gen] = com_project(sqrt_ab_n * x0_pos + sqrt_1mab_n * eps_pos)
        return next_pos, next_x

    def to_absolute(self, pos, x):
        """Frame state -> absolute coordinates and unnormalized type features."""
        pos_abs = pos * self.pos_norm + self.mean.to(pos.device)
        x_abs = (x - self.x_bias) * self.x_norm
        return pos_abs, x_abs


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


class _LigandView:
    """Minimal stand-in for the ligand store of a ``HeteroData``."""

    def __init__(self, pos, x):
        self.pos = pos
        self.x = x


def build_molecules_batched(ops, pos, x, molecule_builder=None):
    """Decode ``[B, N, .]`` frame states into RDKit molecules.

    Returns one ``(mol, smiles, is_valid, is_complete)`` tuple per graph.
    ``MoleculeBuilder`` is always constructed with ``include_invalid=True``:
    with the default it silently drops failures and the returned list stops
    lining up with the reward/value indices.
    """
    if molecule_builder is None:
        molecule_builder = MoleculeBuilder(include_invalid=True)
    pos_abs, x_abs = ops.to_absolute(pos.detach().cpu(), x.detach().cpu())
    results = []
    for i in range(int(pos_abs.shape[0])):
        try:
            mol = molecule_builder.build_mol(_LigandView(pos_abs[i], x_abs[i]))
        except Exception:
            results.append((None, None, False, False))
            continue
        if mol is None:
            results.append((None, None, False, False))
            continue
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            results.append((None, None, False, False))
            continue
        if not smiles:
            results.append((None, None, False, False))
            continue
        results.append((mol, smiles, True, "." not in smiles))
    return results
