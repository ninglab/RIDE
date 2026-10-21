# RIDE: Reference-Anchored Inference-Time Diffusion Editing for Scaffold Hopping

Paper: https://arxiv.org/abs/2609.35623

## Base models

- `configs/conditar.yaml` — conDitar
- `configs/ipdiff.yaml` — IPDiff
- `configs/diffhopp.yaml` — DiffHopp

Select the model for which a checkpoint is available. The three-step pipeline and
every config described below apply identically regardless of which model is
used.

## Repository Layout

- `scripts/` — The command-line entry points for the three pipeline stages described
  below.
- `configs/` — One YAML configuration file per base model.
- `checkpoints/` — Pretrained model weights (see Setup).
- `testset/` — 60 example protein-ligand complexes, each folder
  contains a `_protein.pdb` and a `_ligand.sdf` file.
- `adapters/` — Code that wraps each of the three base models so the pipeline code
  can call them in the same way.
- `ride/` — RIDE's pipeline implementation, corresponding to the three stages
  described below.
- `base_model/` — The unmodified source code for conDitar, IPDiff, and DiffHopp.

## Setup

1. Download the pretrained checkpoints from Google Drive:
   https://drive.google.com/drive/folders/1n0ZtfrMXH5MOkdmOMoVMaeiyvgWbAcQ5?usp=drive_link
2. Place them under `checkpoints/<model>/`, matching the `ckpt:` path in that model's
   config file, e.g.:
   ```
   checkpoints/conditar/diffusion.pt
   checkpoints/ipdiff/ipdiff.pt
   checkpoints/diffhopp/gvp_conditional.ckpt
   ```

## The pipeline

Generating molecules for one protein-ligand complex proceeds in three stages:
reference noise recovery, perturbation segment selection, and value-guided sampling.

### Reference noise recovery

Runs the base model in reverse on the protein (`.pdb`) and its known ligand (`.sdf`)
to recover the noise sequence. This is a one-time step
per complex; the two stages below reuse its output.

```bash
python scripts/invert.py --config configs/conditar.yaml \
    --pdb testset/1dxo/1dxo_protein.pdb \
    --sdf testset/1dxo/1dxo_ligand.sdf \
    --out outputs/1dxo/1dxo_inversion.pt
```

- `--config` — the YAML config file for the base model being used (see Config
  reference below).
- `--pdb` — the protein structure file.
- `--sdf` — the known ligand for that protein.
- `--out` — where to save the result; this file is passed to both later stages.
- `--fixed_pos_npz` (optional) — which atoms to treat as the scaffold to preserve,
  in place of RIDE's default automatic determination from the pocket-ligand interactions.

### Perturbation segment selection

Evaluates several candidate points along the generation process at which to begin
editing the ligand, and reports which one is the best.

```bash
python scripts/select_segment.py --config configs/conditar.yaml \
    --inversion_path outputs/1dxo/1dxo_inversion.pt \
    --out_t1 outputs/1dxo/1dxo_t1.txt \
    --out_csv outputs/1dxo/1dxo_segment_scores.csv
```

- `--config` — same config file used in the previous stage.
- `--inversion_path` — the output of the previous stage.
- `--out_t1` — where to save the selected starting point, to be passed to the next
  stage.
- `--out_csv` (optional) — saves every candidate's score.

### Value-guided sampling

Generates a batch of candidate ligands, starting from the point selected in the
previous stage.

```bash
python scripts/sample.py --config configs/conditar.yaml \
    --inversion_path outputs/1dxo/1dxo_inversion.pt \
    --t1_file outputs/1dxo/1dxo_t1.txt \
    --out outputs/1dxo/1dxo_results/
```

- `--config` — same config file used in the previous stages.
- `--inversion_path` — the output of the first stage.
- `--t1_file` — the starting point selected by the previous stage.
- `--t1` — a starting point given directly, as a plain number, replacing
  `--t1_file` and skipping the previous stage.
- `--out` — a folder; every generate molecule is written there as its own `.sdf`
  file.

## Config

All three config files share the same structure. Check the paper for details on how
these parameters are used.

### Top level

- `model` — base model (`conditar` / `ipdiff` / `diffhopp`).
- `ckpt` — path to the checkpoint file.
- `device` — `cuda` or `cpu`.
- `discrete_forward_sample` — advanced, `conditar`/`ipdiff` only. Controls how atom
  type (a discrete variable) is noised during reference noise recovery: `false`
  (deterministic) or `true` (stochastic). Leave as `false`.

### `t1_grid`

- `N` — number of candidate perturbation starting points.
- `n1`, `n2` — range of candidates to try, out of those `N`.

### `segment_selection`

- `K` — number of generated samples per candidate starting point.
- `2d_weight`, `3d_weight` — scoring weights.

### `value_guided_sampling`

- `M` — number of Monte Carlo samples used for value estimation.
- `num_sample` — number of candidate molecules to be generated per run.
- `value_expand_children` — size of the noise set proposed per intermediate state.
- `value_parent_cap` — max children per intermediate state allowed to survive into
  subsequent generation.
- `value_resample_start`, `value_resample_end` — search range.
- `resample_stride` — search frequency within that range.
- `reward` — reward weights: `2d_weight`/`3d_weight` as above, plus
  `qed_weight` (drug-likeness) and `sa_weight` (synthetic accessibility).
