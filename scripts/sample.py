import argparse
import os

from rdkit import Chem

from ride.perturbation_segment_selection.window import make_window
from ride.value_guided_sampling.reward import RewardWeights
from ride.value_guided_sampling.search import value_guided_search
from ride.utils.metrics import is_disconnected
from ride.utils.trajectory import InversionPayload
from scripts._common import add_model_args, build_adapter, load_config, resolve_complex, set_deterministic


def main():
    set_deterministic()
    p = add_model_args(argparse.ArgumentParser(), require_complex=False)
    p.add_argument('--inversion_path', required=True)
    p.add_argument('--t1', type=int, default=None)
    p.add_argument('--t1_file', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    cfg = load_config(args.config)

    inversion = InversionPayload.load(args.inversion_path)
    pdb, sdf, fixed_pos_npz = resolve_complex(args, inversion)
    adapter = build_adapter(cfg['model'], cfg['ckpt'], cfg.get('device', 'cuda'))
    inversion.to(adapter.device)
    ctx = adapter.load_reference(pdb, sdf, fixed_pos_npz)
    reference_mol = adapter.decode(ctx.state, ctx)

    vgs = cfg['value_guided_sampling']
    reward = vgs.get('reward', {})
    weights = RewardWeights(w_2d=reward.get('2d_weight', 0.0), w_3d=reward.get('3d_weight', 1.0),
                            w_qed=reward.get('qed_weight', 0.0), w_sa=reward.get('sa_weight', 0.0))

    if args.t1 is None and not args.t1_file:
        raise ValueError('need --t1 or --t1_file (see select_segment.py --out_t1)')
    t1 = args.t1 if args.t1 is not None else int(open(args.t1_file).read().strip())

    best_mol, best_reward, survivors = value_guided_search(
        adapter, ctx, inversion, make_window(t1, cfg['L']), reference_mol, weights,
        vgs['value_resample_start'], vgs['value_resample_end'], vgs['resample_stride'],
        vgs['value_expand_children'], vgs['value_parent_cap'],
        M=vgs.get('M', 100), num_sample=vgs.get('num_sample', 1), batch_size=cfg.get('batch_size'))

    print(f'best reward={best_reward:.4f} ({len(survivors)} survivors)')
    os.makedirs(args.out, exist_ok=True)
    pdbid = os.path.basename(pdb).split('.')[0].split('_')[0]
    n_written, n_invalid, n_fragmented = 0, 0, 0
    for mol, reward_i in survivors:
        if mol is None:
            n_invalid += 1
            continue
        if is_disconnected(mol):
            n_fragmented += 1
            continue
        mol.SetProp('reward', f'{reward_i:.8f}')
        mol.SetProp('is_best', str(mol is best_mol))
        writer = Chem.SDWriter(os.path.join(args.out, f'{pdbid}_generated_{n_written}.sdf'))
        writer.write(mol)
        writer.close()
        n_written += 1
    print(f'wrote {n_written}/{len(survivors)} connected molecules to {args.out}/ '
          f'(skipped {n_invalid} invalid, {n_fragmented} fragmented)')


if __name__ == '__main__':
    main()
