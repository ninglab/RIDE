import importlib
import os

import torch
import yaml

_ADAPTER_MODULES = {
    'ipdiff': ('adapters.ipdiff', 'IPDiffAdapter'),
    'conditar': ('adapters.conditar', 'ConditarAdapter'),
    'diffhopp': ('adapters.diffhopp', 'DiffHoppAdapter'),
}


def set_deterministic():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_adapter(model_name: str, ckpt_path: str, device: str = 'cuda',
                  discrete_forward_sample: bool = False):
    if model_name not in _ADAPTER_MODULES:
        raise ValueError(f'--model must be one of {list(_ADAPTER_MODULES)}, got {model_name!r}')
    module_name, class_name = _ADAPTER_MODULES[model_name]
    adapter_cls = getattr(importlib.import_module(module_name), class_name)
    adapter = adapter_cls()
    adapter.load_checkpoint(ckpt_path, device=device)
    adapter.discrete_forward_sample = discrete_forward_sample
    return adapter


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def add_model_args(parser, require_complex=True):
    parser.add_argument('--config', required=True)
    parser.add_argument('--pdb', required=require_complex)
    parser.add_argument('--sdf', required=require_complex)
    parser.add_argument('--fixed_pos_npz')
    return parser


def resolve_complex(args, inversion):
    pdb = args.pdb or inversion.meta.get('pdb')
    sdf = args.sdf or inversion.meta.get('sdf')
    fixed_pos_npz = args.fixed_pos_npz or inversion.meta.get('fixed_pos_npz')
    if not pdb or not sdf:
        raise ValueError('--pdb/--sdf not given and not found in inversion.meta -- '
                         'pass them explicitly or use an inversion written by invert.py')
    return pdb, sdf, fixed_pos_npz
