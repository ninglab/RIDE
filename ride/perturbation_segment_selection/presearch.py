from adapters.base import ModelAdapter, ReferenceContext
from ride.utils.trajectory import InversionPayload
from ride.utils.metrics import sim_2d, sim_3d, is_disconnected
from .grid import t1_grid
from .perturb import perturb
from .window import make_window


def presearch_score(mol, reference_mol, w_2d: float, w_3d: float):
    if mol is None or is_disconnected(mol):
        return None
    d2, d3 = sim_2d(mol, reference_mol), sim_3d(mol, reference_mol)
    return w_2d * d2 + w_3d * (1.0 - d3), d2, d3


def select_t1(adapter: ModelAdapter, ctx: ReferenceContext, inversion: InversionPayload,
             reference_mol, N: int, n1: int, n2: int, L: int, K: int = 1,
             w_2d: float = 0.3, w_3d: float = 0.7, batch_size: int = None):
    results = []
    for t1 in t1_grid(adapter.T, N, n1, n2):
        ctx_b, state, _ = perturb(adapter, ctx, inversion, make_window(t1, L), B=K, label='K',
                                  batch_size=batch_size)
        mols = adapter.decode_batch(state, ctx_b)
        scored = [presearch_score(mol, reference_mol, w_2d, w_3d) for mol in mols]
        valid = [s for s in scored if s is not None]
        n_excluded = len(scored) - len(valid)
        if not valid:
            results.append({'t1': t1, 'score': float('inf'), 'sim_2d': float('nan'),
                            'sim_3d': float('nan'), 'n_excluded': n_excluded})
            continue
        scores, d2s, d3s = zip(*valid)
        n_valid = len(valid)
        results.append({'t1': t1, 'score': sum(scores) / n_valid,
                        'sim_2d': sum(d2s) / n_valid, 'sim_3d': sum(d3s) / n_valid,
                        'n_excluded': n_excluded})
    best_t1 = min(results, key=lambda r: r['score'])['t1']
    return best_t1, results
