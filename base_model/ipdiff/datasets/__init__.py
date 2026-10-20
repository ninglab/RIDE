import torch
from torch.utils.data import Subset
from .pl_pair_dataset import PocketLigandPairDataset


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.get('path', None)   # unused by the prebuilt-lmdb branches
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    elif name == 'pl_scaffold':
        # Prebuilt lmdb carrying ligand_scaffold_mask (+ optional precomputed hbap).
        # The train/valid index split lives in the sidecar meta written at build time.
        from .scaffold_dataset import ScaffoldLigandPairDataset
        dataset = ScaffoldLigandPairDataset(config.lmdb_path, *args, **kwargs)
        meta = torch.load(config.meta_path)
        split = meta['split']

        # Drop complexes whose DecompDiff extraction produced an empty generated
        # region. They are kept in the lmdb for analysis but must not be trained on:
        # generated_mask would be all-False, so the graph contributes exactly zero to
        # the loss while still counting in the per-graph mean, shrinking the gradient.
        # SBDDcode avoids this by resampling in __getitem__; filtering the split here
        # is deterministic and costs no extra I/O.
        # The loss is normalised per graph, so a complex that regenerates 1 atom
        # carries the same weight as one that regenerates 25 -- extreme ratios distort
        # the training distribution in both directions. DecompDiff's own validity
        # window is [0.3, 0.8]; set these to match it once the real distribution is
        # known (meta['ratios'] holds every value, so no rebuild is needed).
        min_ratio = float(config.get('min_scaffold_ratio', 1e-8))
        max_ratio = float(config.get('max_scaffold_ratio', 1.0))
        ratios = meta.get('ratios', None)
        if ratios is not None:
            import numpy as _np
            r = _np.asarray(ratios, dtype=float)
            keep = (r >= min_ratio) & (r <= max_ratio)
            n_before = {k: len(v) for k, v in split.items()}
            split = {k: [i for i in v if keep[i]] for k, v in split.items()}
            dropped = {k: n_before[k] - len(split[k]) for k in split}
            print(f'[pl_scaffold] scaffold_ratio window [{min_ratio}, {max_ratio}]: '
                  f'dropped {dropped}, kept train={len(split["train"])} valid={len(split["valid"])}')
            print(f'[pl_scaffold] ratio distribution over all {len(r)} records: '
                  f'mean={r.mean():.4f} p10={_np.percentile(r,10):.4f} '
                  f'p50={_np.percentile(r,50):.4f} p90={_np.percentile(r,90):.4f} | '
                  f'==0: {int((r==0).sum())}  in[0.3,0.8]: {int(((r>=0.3)&(r<=0.8)).sum())}')

        subsets = {'train': Subset(dataset, indices=split['train']),
                   'valid': Subset(dataset, indices=split['valid'])}
        subsets['test'] = subsets['valid']   # train.py reads subsets['test']
        return dataset, subsets
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    # if 'split' in config:
    #     split = torch.load(config.split)
    #     subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
    #     return dataset, subsets
    # else:
    #     return dataset

    return dataset
