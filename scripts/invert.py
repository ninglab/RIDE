import argparse

from ride.reference_trajectory_recovery.invert import recover_reference_trajectory
from scripts._common import add_model_args, build_adapter, load_config, set_deterministic


def main():
    set_deterministic()
    p = add_model_args(argparse.ArgumentParser())
    p.add_argument('--out', required=True)
    args = p.parse_args()
    cfg = load_config(args.config)

    adapter = build_adapter(cfg['model'], cfg['ckpt'], cfg.get('device', 'cuda'),
                           discrete_forward_sample=cfg.get('discrete_forward_sample', False))
    ctx = adapter.load_reference(args.pdb, args.sdf, args.fixed_pos_npz)
    inversion = recover_reference_trajectory(adapter, ctx)
    inversion.meta.update(model=cfg['model'], pdb=args.pdb, sdf=args.sdf,
                        fixed_pos_npz=args.fixed_pos_npz,
                        discrete_forward_sample=adapter.discrete_forward_sample)
    inversion.save(args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
