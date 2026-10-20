import argparse
import csv
import os

from ride.perturbation_segment_selection.presearch import select_t1
from ride.utils.trajectory import InversionPayload
from scripts._common import add_model_args, build_adapter, load_config, resolve_complex


def main():
    p = add_model_args(argparse.ArgumentParser(), require_complex=False)
    p.add_argument('--inversion_path', required=True)
    p.add_argument('--out_csv', default=None)
    p.add_argument('--out_t1', default=None)
    args = p.parse_args()
    cfg = load_config(args.config)

    inversion = InversionPayload.load(args.inversion_path)
    pdb, sdf, fixed_pos_npz = resolve_complex(args, inversion)
    adapter = build_adapter(cfg['model'], cfg['ckpt'], cfg.get('device', 'cuda'))
    inversion.to(adapter.device)
    ctx = adapter.load_reference(pdb, sdf, fixed_pos_npz)
    reference_mol = adapter.decode(ctx.state, ctx)

    t1_grid, segment_selection = cfg['t1_grid'], cfg.get('segment_selection', {})
    best_t1, results = select_t1(
        adapter, ctx, inversion, reference_mol, t1_grid['N'], t1_grid['n1'], t1_grid['n2'], cfg['L'],
        K=segment_selection.get('K', 1), w_2d=segment_selection.get('2d_weight', 0.3),
        w_3d=segment_selection.get('3d_weight', 0.7), batch_size=cfg.get('batch_size'))

    print(f'selected t1={best_t1}')
    for r in sorted(results, key=lambda r: r['score']):
        print(f'  t1={r["t1"]:<5d} score={r["score"]:.4f} '
              f'sim_2d={r["sim_2d"]:.3f} sim_3d={r["sim_3d"]:.3f} n_excluded={r["n_excluded"]}')
    for out_path in (args.out_csv, args.out_t1):
        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['t1', 'score', 'sim_2d', 'sim_3d', 'n_excluded'])
            w.writeheader()
            w.writerows(results)
    if args.out_t1:
        with open(args.out_t1, 'w') as fh:
            fh.write(str(best_t1))


if __name__ == '__main__':
    main()
