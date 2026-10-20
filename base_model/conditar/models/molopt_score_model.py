"""
the newest version about how to incorporate shape embedding with electrostatic features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean, scatter_softmax, scatter_min
import numpy as np
from tqdm.auto import tqdm
import random
from models.common import ShiftedSoftplus, MLP, GaussianSmearing, find_closest_points, compute_bond_angle, compute_torsion_angle
from models.uni_transformer import UniTransformerO2TwoUpdateGeneralWeightedPoolDynamicUpdateGVPLateBiLevelSeperate
from models.diffusion import *
from sklearn.neighbors import KDTree
import pdb
import abc
# import pytorch_lightning as pl
import utils.train as utils_train

import utils.transforms as trans
from utils import misc, reconstruct
from scipy.optimize import linear_sum_assignment


def get_refine_net(config):
    refine_net = UniTransformerO2TwoUpdateGeneralWeightedPoolDynamicUpdateGVPLateBiLevelSeperate
    
    refine_net = refine_net(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            scalar_hidden_dim=config.scalar_hidden_dim,
            vec_hidden_dim=config.vec_hidden_dim,
            shape_dim=config.shape_dim,
            shape_latent_dim=config.shape_latent_dim,
            n_heads=config.n_heads,
            k=config.knn,
            pocket_k=config.pocket_knn,
            residue_k=config.residue_knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            r_feat_mode=config.r_feat_mode,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            atom_enc_mode=config.atom_enc_mode,
            sync_twoup=config.sync_twoup,
            pred_bond_type=config.pred_bond_type,
            use_shape_vec_mul=config.use_shape_vec_mul,
            use_residue=config.use_residue,
            time_emb_dim=config.time_emb_dim,
            residue_pooling=config.residue_pooling
    )
    return refine_net

def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x
                              
def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)

def center_pos(ligand_pos, batch_ligand, mode='none'):
    if mode == 'none':
        offset = 0.0
    elif mode == 'center':
        offset = scatter_mean(ligand_pos, batch_ligand, dim=0)
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return ligand_pos, offset


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits, gumbel_noise=None):
    if gumbel_noise is None:
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

# %%

def dynamic_threshold(x0, p):
    s = torch.quantile(x0, p)
    x0 = torch.clip(x0, min=-s, max=s)
    return x0

def reference_threshold(x0, x0_cond, p):
    s = torch.max(torch.abs(x0_cond)) * p
    x0 = torch.clip(x0, min=-s, max=s)
    return x0

def rescale(x0, x0_cond, p):
    std_x0 = torch.std(x0)
    std_x0_cond = torch.std(x0_cond)
    rescale_ratio = std_x0_cond / std_x0
    x0_rescale = x0 * rescale_ratio
    x0 = p * x0_rescale + (1-p) * x0
    return x0

def threshold_CFG(x0, x0_cond, threshold_type, threshold_args, bounds=None):
    if threshold_type == 'reference_threshold':
        p = threshold_args.get('p', 1.1)
        x0 = reference_threshold(x0, x0_cond, p)
    elif threshold_type == 'dynamic_threshold':
        p = threshold_args.get('p', 0.995)
        x0 = dynamic_threshold(x0, p)
    elif threshold_type == 'rescale':
        p = threshold_args.get('p', 0.7)
        x0 = rescale(x0, x0_cond, p)
    elif threshold_type is not None:
        raise ValueError("undefined thresholding strategy: expect one of (reference_threshold, dynamic_threshold, rescale, none) " + \
                         "but get %s" % (threshold_type))
    
    if bounds is not None:
        x0 = torch.clamp(x0, min=bounds[:,0], max=bounds[:, 1])
    return x0

# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
        

# Model
class ScorePosNet3D(nn.Module):

    def __init__(self, config, ligand_atom_feature_dim, ligand_bond_feature_dim):
        # super().__init__(config.train)
        super().__init__()
        self.config = config.model

        # variance schedule
        self.denoise_type = self.config.denoise_type  # ['diffusion', 'score_matching']
        self.model_mean_type = self.config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = self.config.loss_v_weight
        self.loss_weight_type = self.config.loss_weight_type

        self.v_mode = self.config.v_mode
        self.v_net_type = getattr(self.config, 'v_net_type', 'mlp')

        self.pred_bond_type = self.config.pred_bond_type
        if self.pred_bond_type:
            self.loss_bond_type = nn.CrossEntropyLoss(reduce=False)
            self.loss_bond_weight = self.config.loss_bond_weight

        self.use_bond_dist_loss = self.config.use_bond_dist_loss
        self.loss_bond_dist_weight = self.config.loss_bond_dist_weight
        self.use_bond_angle_loss = self.config.use_bond_angle_loss
        self.loss_bond_angle_weight = self.config.loss_bond_angle_weight
        self.use_torsion_angle_loss = self.config.use_torsion_angle_loss
        self.loss_torsion_angle_weight = self.config.loss_torsion_angle_weight
        self.loss_torsion_angle_type = self.config.loss_torsion_angle_type

        self.sample_time_method = self.config.sample_time_method  # ['importance', 'symmetric']
        self.loss_pos_type = self.config.loss_pos_type  # ['mse', 'kl']
        # print(f'Loss pos mode {self.loss_pos_type} applied!')

        betas = get_beta_schedule(
            num_diffusion_timesteps=self.config.num_diffusion_timesteps,
            **self.config.schedule_pos
        )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        if self.loss_weight_type == 'noise_level':
            snr_values = alphas_cumprod / (1-alphas_cumprod)
            self.loss_pos_step_weight = to_torch_const(np.clip(self.config.loss_pos_min_weight + snr_values, None, self.config.loss_pos_max_weight))

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        # self.posterior_logvar = to_torch_const(np.log(np.maximum(posterior_variance, 1e-10)))
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        betas_v = get_beta_schedule(
            num_diffusion_timesteps=self.config.num_diffusion_timesteps,
            **self.config.schedule_v
        )
        alphas_v = 1. - betas_v

        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        # model definition
        self.scalar_hidden_dim = self.config.scalar_hidden_dim
        self.vec_hidden_dim = self.config.vec_hidden_dim
        self.num_classes = ligand_atom_feature_dim
        
        ###### to test ######
        # center pos
        self.center_pos_mode = self.config.center_pos_mode  # ['none', 'center']

        # time embedding
        self.time_emb_dim = self.config.time_emb_dim
        if self.time_emb_dim > 0:
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 2),
                nn.SiLU(),
                nn.Linear(self.time_emb_dim * 2, self.time_emb_dim)
            )
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim + int(self.v_mode=='tomask'), self.scalar_hidden_dim)
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + int(self.v_mode=='tomask'), self.scalar_hidden_dim)
            
        self.refine_net = get_refine_net(self.config)

        # print(f'v net type: {self.v_net_type}')
        if self.v_net_type == 'mlp':
            self.v_inference = nn.Sequential(
                nn.Linear(self.scalar_hidden_dim, self.scalar_hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.scalar_hidden_dim, ligand_atom_feature_dim + int(self.v_mode=='tomask')),
            )
        elif self.v_net_type == 'attention':
            self.v_distance_expansion = GaussianSmearing(0., 10., num_gaussians=self.config.num_r_gaussian)
            norm = True
            act_fn = 'relu'
            kv_input_dim = self.scalar_hidden_dim * 2 + self.config.num_r_gaussian
            self.v_n_heads = 16
            self.vk_func = MLP(kv_input_dim, self.scalar_hidden_dim, self.scalar_hidden_dim, norm=norm, act_fn=act_fn)
            self.vv_func = MLP(kv_input_dim, self.scalar_hidden_dim, self.scalar_hidden_dim, norm=norm, act_fn=act_fn)
            self.vq_func = MLP(self.scalar_hidden_dim, self.scalar_hidden_dim, self.scalar_hidden_dim, norm=norm, act_fn=act_fn)
            self.v_inference = nn.Sequential(
                nn.Linear(self.scalar_hidden_dim * 2, self.scalar_hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.scalar_hidden_dim, ligand_atom_feature_dim + int(self.v_mode=='tomask')),
            )
        else:
            raise NotImplementedError
        self.cond_mask_prob = self.config.cond_mask_prob

        self.optimizer = utils_train.get_optimizer(config.train.optimizer, self)
        self.scheduler = utils_train.get_scheduler(config.train.scheduler, self.optimizer)

    def forward(self, ligand_pos_perturbed, ligand_v_perturbed, 
                batch_ligand, pocket_data, time_step=None, ligand_bond_index=None, ligand_shape=None,
                mask_shape_emb=None, ligand_bond_type=None, pred_bond=True, if_test=False, return_all=False,
                pca_perturb=None):
        """
        f(x0, v0 | xt, vt): predicts the original position and atom type from noisy samples at step t
        """
        batch_size = batch_ligand.max().item() + 1

        ligand_v_perturbed = F.one_hot(ligand_v_perturbed, self.num_classes+int(self.v_mode=='tomask')).float()
    
        # time embedding
        if self.time_emb_dim > 0:
            time_feat = self.time_emb(time_step)[batch_ligand]
            ligand_feat = torch.cat([ligand_v_perturbed, time_feat], -1)
        else:
            ligand_feat = ligand_v_perturbed
            time_feat = None

        ligand_emb = self.ligand_atom_emb(ligand_feat)
        
        outputs = self.refine_net(ligand_v_perturbed, ligand_emb, ligand_pos_perturbed, batch_ligand, 
                                  pocket_data, ligand_bond_index=ligand_bond_index, ligand_shape=ligand_shape, mask_shape_emb=mask_shape_emb,
                                  ligand_bond_type=ligand_bond_type, pred_bond=pred_bond, if_test=if_test, return_all=return_all,
                                  pca_perturb=pca_perturb)
        final_pos, final_h = outputs['x'], outputs['h']
        final_v = self.v_inference(final_h)

        preds = {
            'pred_ligand_pos': final_pos,
            'pred_ligand_h': final_h,
            'pred_ligand_v': final_v,
        }
        if self.pred_bond_type and pred_bond:
            preds.update({
                'bond_pred': outputs['bond_pred'],
                'bond_gt': outputs['bond_gt'],
                'edge_index': outputs['edge_index'],
                'all_bond_loss': outputs['all_bond_loss']
            })

        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds


    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        """
        forward diffusion process: q(vt | vt-1)
        """
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        if self.v_mode == 'uniform':
            # alpha_t * vt-1 + (1 - alpha_t) 1 / K
            log_probs = log_add_exp(
                log_vt_1 + log_alpha_t,
                log_1_min_alpha_t - np.log(self.num_classes)
            )
        elif self.v_mode == 'tomask':
            # alpha_t * vt for category at step t and (1 - alpha_t) for mask
            log_probs = log_vt_1 + log_alpha_t
            log_probs[:, -1] = log_1_min_alpha_t.squeeze(1)
        else:
            raise ValueError("undefined v_mode: %s (expect uniform or tomask)" % (self.v_mode))  
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        """
        forward diffusion process: q(vt | v0)
        """
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        if self.v_mode == 'uniform':
            # cum_alpha_t * v0 + (1 - cum_alpha_t) 1 / K
            log_probs = log_add_exp(
                log_v0 + log_cumprod_alpha_t,
                log_1_min_cumprod_alpha - np.log(self.num_classes)
            )
        elif self.v_mode == 'tomask':
            # cum_alpha_t * v0 
            # (1 - cum_alpha_t) for mask
            log_probs = log_v0 + log_cumprod_alpha_t
            log_probs[:, -1] = log_1_min_cumprod_alpha.squeeze(1)
        else:
            raise ValueError("undefined v_mode: %s (expect uniform or tomask)" % (self.v_mode))
        return log_probs

    def q_v_sample(self, log_v0, t, batch, num_classes):
        """
        backward generative process q(vt)
        """
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, num_classes)
        
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(pos0)
        pos_noise.normal_()
        pos_perturbed = a_pos.sqrt() * pos0 + (1.0 - a_pos).sqrt() * pos_noise
        pos_prior = torch.randn_like(pos_perturbed)
        kl_prior = torch.mean((pos_perturbed - pos_prior) ** 2)
        return kl_prior

    def sample_time(self, num_graphs, device):
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        # print('kl pos: ', kl_pos, 'nll pos: ', decoder_nll_pos, 'loss pos: ', loss_pos)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        # print('kl v: ', kl_v, 'nll v: ', decoder_nll_v, 'loss v: ', loss_v)
        return loss_v

    def compute_bond_angle_loss(self, n0_dst, pred_pos, gt_pos, bond_index, batch_bond):
        src, dst = bond_index

        # Calculate ground truth angles
        gt_pos_ji = gt_pos[src] - gt_pos[dst]
        gt_pos_j0 = gt_pos[n0_dst] - gt_pos[dst]

        gt_angle = compute_bond_angle(gt_pos_ji, gt_pos_j0)

        # Calculate predicted angles
        pred_pos_ji = pred_pos[src] - pred_pos[dst]
        pred_pos_j0 = pred_pos[n0_dst] - pred_pos[dst]

        pred_angle = compute_bond_angle(pred_pos_ji, pred_pos_j0)
        
        angle_mse = scatter_mean(((gt_angle - pred_angle) ** 2), batch_bond, dim=0)
        return angle_mse

    def compute_torsion_angle_loss(self, n0_src, n1_src, n0_dst, n1_dst, pred_pos, gt_pos, bond_index, batch_bond, torsion_type='one'):
        src, dst = bond_index

        gt_pos_ji =  gt_pos[dst] - gt_pos[src]
        pred_pos_ji = pred_pos[dst] - pred_pos[src]
        """
        # Calculate ground truth angles
        gt_pos_ji = gt_pos[src] - gt_pos[dst]
        gt_pos_j0 = gt_pos[n0_dst] - gt_pos[dst]
        gt_pos_j1 = gt_pos[n1_dst] - gt_pos[dst]
        gt_torsion_angle = compute_torsion_angle(gt_pos_ji, gt_pos_j0, gt_pos_j1)

        # Calculate predicted angles
        pred_pos_ji = pred_pos[src] - pred_pos[dst]
        pred_pos_j0 = pred_pos[n0_dst] - pred_pos[dst]
        pred_pos_j1 = pred_pos[n1_dst] - pred_pos[dst]
        pred_torsion_angle = compute_torsion_angle(pred_pos_ji, pred_pos_j0, pred_pos_j1)
        #print(gt_torsion_angle)
        #print(pred_torsion_angle)
        #gt_torsion_angle - - pred_torsion_angle
        torsion_angle_mse = scatter_mean(((gt_torsion_angle - pred_torsion_angle) ** 2), batch_bond, dim=0)
        #print(torsion_angle_mse)
        """
        # Calculate ground truth angles
        src_ref_mask = n0_src == dst
        src_ref = torch.clone(n0_src)
        src_ref[src_ref_mask] = n1_src[src_ref_mask]
        
        dst_ref_mask = n0_dst == src
        dst_ref = torch.clone(n0_dst)
        dst_ref[dst_ref_mask] = n1_dst[dst_ref_mask]
        
        gt_pos_src_ref = gt_pos[src_ref] - gt_pos[dst]
        gt_pos_dst_ref = gt_pos[dst_ref] - gt_pos[dst]
        gt_torsion_angle1 = compute_torsion_angle(gt_pos_ji, gt_pos_src_ref, gt_pos_dst_ref)

        pred_pos_src_ref = pred_pos[src_ref] - pred_pos[dst]
        pred_pos_dst_ref = pred_pos[dst_ref] - pred_pos[dst]
        pred_torsion_angle1 = compute_torsion_angle(pred_pos_ji, pred_pos_src_ref, pred_pos_dst_ref)
        
        angle_diff_1 = (gt_torsion_angle1 - pred_torsion_angle1)
        angle_diff_2 = 2 * torch.pi - (gt_torsion_angle1 - pred_torsion_angle1)
        angle_diff, _ = torch.min(torch.abs(torch.stack([angle_diff_1, angle_diff_2], dim=1)), dim=1)
        torsion_angle_mse = scatter_mean((angle_diff ** 2), batch_bond, dim=0)
        #torsion_angle_mse = (torsion_angle_mse + torsion_angle_mse1) / 2
        
        return torsion_angle_mse

    def get_diffusion_loss(self, ligand_data, pocket_data, time_step=None, eval_mode=False, batch_weights=None):
        # print(ligand_data.ligand_element)
        batch_ligand = ligand_data.batch
        ligand_pos = ligand_data.ligand_coordinate
        ligand_v = ligand_data.ligand_atom_feature_full
        ligand_bond_index = ligand_data.ligand_bond_index
        ligand_bond_type = ligand_data.ligand_bond_type
        
        num_graphs = batch_ligand.max().item() + 1 
        ligand_pos, _ = center_pos(ligand_pos, batch_ligand, mode=self.center_pos_mode)
        
        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, ligand_pos.device)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)
        
        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        if self.v_mode == 'uniform':
            # Vt = a * V0 + (1-a) / K
            log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
            ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand, self.num_classes)
        elif self.v_mode == 'tomask':
            # Vt = a * V0 + (1-a) * mask
            log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes+1)
            ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand, self.num_classes+1)
        else:
            raise ValueError
        
        ligand_shape = None
        mask_shape_emb = None

        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            ligand_pos_perturbed=ligand_pos_perturbed.float(),
            ligand_v_perturbed=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            pocket_data=pocket_data,
            ligand_shape=ligand_shape,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
            mask_shape_emb=mask_shape_emb,
            time_step=time_step,
            pred_bond=True
        )
        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        
        # atom type
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        
        
        loss_pos_weight = None
        if self.loss_weight_type == 'noise_level':
            loss_pos_weight = self.loss_pos_step_weight.index_select(0, time_step)
            
        # unweighted
        target, pred = ligand_pos, pred_ligand_pos
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        
        if batch_weights is not None:
            loss_pos = loss_pos * batch_weights
            kl_v = kl_v * batch_weights

        if self.loss_weight_type == 'uniform':
            loss_pos = torch.mean(loss_pos)
        elif self.loss_weight_type == 'noise_level':
            loss_pos = torch.mean(loss_pos_weight * loss_pos)
        
        loss_v = torch.mean(kl_v)
        
        loss = loss_pos + loss_v * self.loss_v_weight

        if self.pred_bond_type:
            pred_bond_losses = torch.stack(preds['all_bond_loss'], dim=1)
            pred_bond_types, gt_bond_types = preds['bond_pred'], preds['bond_gt']

            if batch_weights is not None:
                pred_bond_losses = batch_weights.unsqueeze(1) * pred_bond_losses

            if self.loss_weight_type == 'noise_level':
                loss_bond_all = loss_pos_weight.unsqueeze(1) * pred_bond_losses

            loss_bond_aux = torch.mean(loss_bond_all[:, :-1])
            loss_bond_final = torch.mean(loss_bond_all[:, -1])
            loss += self.loss_bond_weight * (loss_bond_aux + loss_bond_final)
        else:
            loss_bond_aux, loss_bond_final, pred_bond_types, gt_bond_types = 0.0, 0.0, [], []
        
        pred_bond_dist = get_distance(pred_ligand_pos, ligand_bond_index)
        gt_bond_dist = get_distance(target, ligand_bond_index)
        batch_ligand_bond = torch.index_select(batch_ligand, 0, ligand_bond_index[0, :])
        loss_bond_dist = scatter_mean(((pred_bond_dist - gt_bond_dist) ** 2), batch_ligand_bond, dim=0)
        if batch_weights is not None:
            loss_bond_dist = batch_weights * loss_bond_dist
        
        if self.loss_weight_type == 'noise_level':
            loss_bond_dist = loss_pos_weight * loss_bond_dist
        loss_bond_dist = torch.mean(loss_bond_dist)

        if self.use_bond_dist_loss:
            loss += self.loss_bond_dist_weight * loss_bond_dist
        #else:
        #    pred_bond_dist, gt_bond_dist, loss_bond_dist = None, None, 0.0

        src, dst = ligand_bond_index
        #pdb.set_trace()      
        n0, n1 = find_closest_points(gt_bond_dist, ligand_bond_index)
        
        n0_dst = torch.index_select(n0, 0, dst)
        n1_dst = torch.index_select(n1, 0, dst)

        loss_bond_angle = self.compute_bond_angle_loss(n0_dst, pred_ligand_pos, target, ligand_bond_index, batch_ligand_bond)
        if self.loss_weight_type == 'noise_level':
            loss_bond_angle = loss_pos_weight * loss_bond_angle
        loss_bond_angle = torch.mean(loss_bond_angle)

        n0_src = torch.index_select(n0, 0, src)
        n1_src = torch.index_select(n1, 0, src)

        loss_torsion_angle = self.compute_torsion_angle_loss(n0_src, n1_src, n0_dst, n1_dst, pred_ligand_pos, target, ligand_bond_index, 
                                                        batch_ligand_bond, torsion_type=self.loss_torsion_angle_type)
        
        if self.loss_weight_type == 'noise_level':
            loss_torsion_angle = loss_pos_weight * loss_torsion_angle
        loss_torsion_angle = torch.mean(loss_torsion_angle)

        if self.use_bond_angle_loss:
            loss += self.loss_bond_angle_weight * loss_bond_angle
            
        if self.use_torsion_angle_loss:
            loss += self.loss_torsion_angle_weight * loss_torsion_angle
        
        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_bond_aux': loss_bond_aux,
            'loss_bond_final': loss_bond_final,
            'loss_bond_dist': loss_bond_dist,
            'loss_bond_angle': loss_bond_angle,
            'loss_torsion_angle': loss_torsion_angle,
            'loss': loss,
            'x0': ligand_pos,
            'ligand_pos_perturbed': ligand_pos_perturbed,
            'ligand_v_perturbed': ligand_v_perturbed,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_bond_type': pred_bond_types,
            'gt_bond_type': gt_bond_types,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)
        }


    def get_diffusion_loss_scaffold(self, ligand_data, pocket_data, time_step=None, eval_mode=False, batch_weights=None):
        batch_ligand = ligand_data.batch
        ligand_pos = ligand_data.ligand_coordinate
        ligand_v = ligand_data.ligand_atom_feature_full
        ligand_bond_index = ligand_data.ligand_bond_index
        ligand_bond_type = ligand_data.ligand_bond_type

        scaffold_mask = None
        if hasattr(ligand_data, 'ligand_scaffold_mask'):
            scaffold_mask = ligand_data.ligand_scaffold_mask
        elif hasattr(ligand_data, 'ligand_scaffold'):
            scaffold_mask = ligand_data.ligand_scaffold
        if scaffold_mask is None:
            raise ValueError('get_diffusion_loss_scaffold requires ligand_scaffold_mask or ligand_scaffold in ligand_data.')

        scaffold_mask = torch.as_tensor(scaffold_mask, device=ligand_pos.device).view(-1)
        if scaffold_mask.numel() != ligand_pos.size(0):
            # Mismatch likely due to hydrogen filtering; fall back to standard diffusion loss.
            print('scaffold mask and ligand size mismatch')
            return self.get_diffusion_loss(
                ligand_data=ligand_data,
                pocket_data=pocket_data,
                time_step=time_step,
                eval_mode=eval_mode,
                batch_weights=batch_weights,
            )
        # Train with noisy scaffold and clean unscaffold conditioning.
        generated_mask = scaffold_mask == 1
        fixed_mask = ~generated_mask

        num_graphs = batch_ligand.max().item() + 1
        ligand_pos, _ = center_pos(ligand_pos, batch_ligand, mode=self.center_pos_mode)

        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, ligand_pos.device)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)

        a_pos = a[batch_ligand].unsqueeze(-1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        ligand_pos_noisy = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise
        ligand_pos_perturbed = ligand_pos.clone()
        ligand_pos_perturbed[generated_mask] = ligand_pos_noisy[generated_mask]

        if self.v_mode == 'uniform':
            log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
            ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand, self.num_classes)
            log_ligand_v_clean = log_ligand_v0
        elif self.v_mode == 'tomask':
            log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes + 1)
            ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand, self.num_classes + 1)
            log_ligand_v_clean = log_ligand_v0
        else:
            raise ValueError

        ligand_v_perturbed = ligand_v_perturbed.clone()
        ligand_v_perturbed[fixed_mask] = ligand_v[fixed_mask]
        log_ligand_vt = log_ligand_vt.clone()
        log_ligand_vt[fixed_mask] = log_ligand_v_clean[fixed_mask]

        ligand_shape = None
        mask_shape_emb = None

        preds = self(
            ligand_pos_perturbed=ligand_pos_perturbed.float(),
            ligand_v_perturbed=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            pocket_data=pocket_data,
            ligand_shape=ligand_shape,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
            mask_shape_emb=mask_shape_emb,
            time_step=time_step,
            pred_bond=self.pred_bond_type,
        )
        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']

        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)

        generated_float = generated_mask.float()
        generated_count = scatter_sum(generated_float, batch_ligand, dim=0).clamp_min(1.0)

        loss_pos_weight = None
        if self.loss_weight_type == 'noise_level':
            loss_pos_weight = self.loss_pos_step_weight.index_select(0, time_step)

        target, pred = ligand_pos, pred_ligand_pos
        loss_pos_atom = ((pred - target) ** 2).sum(-1) * generated_float
        loss_pos = scatter_sum(loss_pos_atom, batch_ligand, dim=0) / generated_count

        kl_v_atom = categorical_kl(log_v_true_prob, log_v_model_prob)
        decoder_nll_v = -log_categorical(log_ligand_v0, log_v_model_prob)
        t0_mask = (time_step == 0).float()[batch_ligand]
        loss_v_atom = (t0_mask * decoder_nll_v + (1.0 - t0_mask) * kl_v_atom) * generated_float
        kl_v = scatter_sum(loss_v_atom, batch_ligand, dim=0) / generated_count

        if batch_weights is not None:
            loss_pos = loss_pos * batch_weights
            kl_v = kl_v * batch_weights

        if self.loss_weight_type == 'uniform':
            loss_pos = torch.mean(loss_pos)
        elif self.loss_weight_type == 'noise_level':
            loss_pos = torch.mean(loss_pos_weight * loss_pos)

        loss_v = torch.mean(kl_v)
        loss = loss_pos + loss_v * self.loss_v_weight

        # For interface bond supervision, keep the generated part predicted and replace the
        # fixed scaffold part with the ground-truth clean values.
        pred_ligand_pos_for_bond = pred_ligand_pos.clone()
        pred_ligand_pos_for_bond[fixed_mask] = ligand_pos[fixed_mask]

        pred_ligand_v_for_bond = pred_ligand_v.clone()
        pred_ligand_v_for_bond[fixed_mask] = -1.0e5
        pred_ligand_v_for_bond[fixed_mask, ligand_v[fixed_mask]] = 1.0e5

        src, dst = ligand_bond_index
        interface_bond_mask = torch.logical_xor(fixed_mask[src], fixed_mask[dst])
        batch_ligand_bond = torch.index_select(batch_ligand, 0, src)
        interface_bond_count = scatter_sum(interface_bond_mask.float(), batch_ligand_bond, dim=0).clamp_min(1.0)

        zero = loss.new_zeros(())
        pred_bond_types, gt_bond_types = [], []
        loss_bond_aux = zero
        loss_bond_final = zero
        loss_bond_dist = zero
        loss_bond_angle = zero
        loss_torsion_angle = zero

        if self.pred_bond_type and preds.get('bond_pred', None) is not None and preds.get('bond_gt', None) is not None:
            pred_bond_types, gt_bond_types = preds['bond_pred'], preds['bond_gt']
            pred_edge_index = preds['edge_index']
            pred_src, pred_dst = pred_edge_index
            pred_interface_mask = torch.logical_xor(fixed_mask[pred_src], fixed_mask[pred_dst])
            if torch.any(pred_interface_mask):
                bond_ce = F.cross_entropy(pred_bond_types, gt_bond_types, reduction='none')
                bond_ce = bond_ce * pred_interface_mask.float()
                batch_pred_edge = torch.index_select(batch_ligand, 0, pred_src)
                pred_interface_count = scatter_sum(pred_interface_mask.float(), batch_pred_edge, dim=0).clamp_min(1.0)
                loss_bond_final = scatter_sum(bond_ce, batch_pred_edge, dim=0) / pred_interface_count
                if batch_weights is not None:
                    loss_bond_final = loss_bond_final * batch_weights
                if self.loss_weight_type == 'noise_level':
                    loss_bond_final = torch.mean(loss_pos_weight * loss_bond_final)
                else:
                    loss_bond_final = torch.mean(loss_bond_final)
                loss += self.loss_bond_weight * loss_bond_final

        pred_bond_dist = get_distance(pred_ligand_pos_for_bond, ligand_bond_index)
        gt_bond_dist = get_distance(target, ligand_bond_index)
        bond_dist_sq = ((pred_bond_dist - gt_bond_dist) ** 2) * interface_bond_mask.float()
        loss_bond_dist = scatter_sum(bond_dist_sq, batch_ligand_bond, dim=0) / interface_bond_count
        if batch_weights is not None:
            loss_bond_dist = batch_weights * loss_bond_dist
        if self.loss_weight_type == 'noise_level':
            loss_bond_dist = loss_pos_weight * loss_bond_dist
        loss_bond_dist = torch.mean(loss_bond_dist)
        if self.use_bond_dist_loss:
            loss += self.loss_bond_dist_weight * loss_bond_dist

        n0, n1 = find_closest_points(gt_bond_dist, ligand_bond_index)
        n0_dst = torch.index_select(n0, 0, dst)
        n1_dst = torch.index_select(n1, 0, dst)

        gt_pos_ji = target[src] - target[dst]
        gt_pos_j0 = target[n0_dst] - target[dst]
        gt_angle = compute_bond_angle(gt_pos_ji, gt_pos_j0)
        pred_pos_ji = pred_ligand_pos_for_bond[src] - pred_ligand_pos_for_bond[dst]
        pred_pos_j0 = pred_ligand_pos_for_bond[n0_dst] - pred_ligand_pos_for_bond[dst]
        pred_angle = compute_bond_angle(pred_pos_ji, pred_pos_j0)
        bond_angle_sq = ((gt_angle - pred_angle) ** 2) * interface_bond_mask.float()
        loss_bond_angle = scatter_sum(bond_angle_sq, batch_ligand_bond, dim=0) / interface_bond_count
        if self.loss_weight_type == 'noise_level':
            loss_bond_angle = loss_pos_weight * loss_bond_angle
        loss_bond_angle = torch.mean(loss_bond_angle)
        if self.use_bond_angle_loss:
            loss += self.loss_bond_angle_weight * loss_bond_angle

        n0_src = torch.index_select(n0, 0, src)
        n1_src = torch.index_select(n1, 0, src)
        src_ref_mask = n0_src == dst
        src_ref = torch.clone(n0_src)
        src_ref[src_ref_mask] = n1_src[src_ref_mask]
        dst_ref_mask = n0_dst == src
        dst_ref = torch.clone(n0_dst)
        dst_ref[dst_ref_mask] = n1_dst[dst_ref_mask]

        gt_pos_ji_t = target[dst] - target[src]
        gt_pos_src_ref = target[src_ref] - target[dst]
        gt_pos_dst_ref = target[dst_ref] - target[dst]
        gt_torsion_angle = compute_torsion_angle(gt_pos_ji_t, gt_pos_src_ref, gt_pos_dst_ref)

        pred_pos_ji_t = pred_ligand_pos_for_bond[dst] - pred_ligand_pos_for_bond[src]
        pred_pos_src_ref = pred_ligand_pos_for_bond[src_ref] - pred_ligand_pos_for_bond[dst]
        pred_pos_dst_ref = pred_ligand_pos_for_bond[dst_ref] - pred_ligand_pos_for_bond[dst]
        pred_torsion_angle = compute_torsion_angle(pred_pos_ji_t, pred_pos_src_ref, pred_pos_dst_ref)
        angle_diff_1 = gt_torsion_angle - pred_torsion_angle
        angle_diff_2 = 2 * torch.pi - (gt_torsion_angle - pred_torsion_angle)
        angle_diff, _ = torch.min(torch.abs(torch.stack([angle_diff_1, angle_diff_2], dim=1)), dim=1)
        torsion_sq = (angle_diff ** 2) * interface_bond_mask.float()
        loss_torsion_angle = scatter_sum(torsion_sq, batch_ligand_bond, dim=0) / interface_bond_count
        if self.loss_weight_type == 'noise_level':
            loss_torsion_angle = loss_pos_weight * loss_torsion_angle
        loss_torsion_angle = torch.mean(loss_torsion_angle)
        if self.use_torsion_angle_loss:
            loss += self.loss_torsion_angle_weight * loss_torsion_angle

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_bond_aux': loss_bond_aux,
            'loss_bond_final': loss_bond_final,
            'loss_bond_dist': loss_bond_dist,
            'loss_bond_angle': loss_bond_angle,
            'loss_torsion_angle': loss_torsion_angle,
            'loss': loss,
            'x0': ligand_pos,
            'ligand_pos_perturbed': ligand_pos_perturbed,
            'ligand_v_perturbed': ligand_v_perturbed,
            'pred_ligand_pos': pred_ligand_pos_for_bond,
            'pred_ligand_v': pred_ligand_v_for_bond,
            'pred_bond_type': pred_bond_types,
            'gt_bond_type': gt_bond_types,
            'ligand_v_recon': F.softmax(pred_ligand_v_for_bond, dim=-1),
            'generated_mask': generated_mask,
            'fixed_mask': fixed_mask,
        }

    @torch.no_grad()
    def sample_diffusion(self, init_ligand_pos, init_ligand_v, batch_ligand, pocket_data, ligand_shape=None, 
                         mask_shape_emb=None, num_steps=None, center_pos_mode=None, pred_bond=False):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_ligand.max().item() + 1
        # print('sample center pos mode: ', center_pos_mode)

        init_ligand_pos, offset = center_pos(init_ligand_pos, batch_ligand, mode=center_pos_mode)
        
        pos_traj, v_traj = [], []
        pos_cond_traj, v_cond_traj = [], []
        pos_uncond_traj, v_uncond_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        pred_bonds = None
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
        # for i in time_seq:
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)
            with torch.no_grad():
                preds_with_cond = self(
                    ligand_pos_perturbed=ligand_pos,
                    ligand_v_perturbed=ligand_v,
                    batch_ligand=batch_ligand,
                    pocket_data=pocket_data,
                    ligand_shape=ligand_shape,
                    mask_shape_emb=mask_shape_emb,
                    if_test=True,
                    pred_bond=True if self.pred_bond_type and pred_bond and i == 0 else False,
                    time_step=t
                )

            if self.pred_bond_type and pred_bond and i == 0:
                pred_bonds = torch.argmax(preds_with_cond['bond_pred'], dim=-1).view(-1, 1)
                pred_edge_indices = preds_with_cond['edge_index'].transpose(0, 1)
                pred_bonds = torch.cat([pred_edge_indices, pred_bonds], dim=1)

            preds = preds_with_cond

            pos_cond_traj.append(preds_with_cond['pred_ligand_pos'])
            v_cond_traj.append(preds_with_cond['pred_ligand_v'])
            
            if self.v_mode == 'tomask':
                #mask the mask atom
                preds['pred_ligand_v'][:, -1] = -1.e5

            # Compute posterior mean and variance
            with torch.no_grad():
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
                
                pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                # no noise when t == 0
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)

                ligand_pos = ligand_pos_next

                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes+int(self.v_mode=='tomask'))
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

                if center_pos_mode != 'none':
                    ori_ligand_pos = ligand_pos + offset[batch_ligand]
                else:
                    ori_ligand_pos = ligand_pos
                
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())
    
        # if center_pos_mode != 'none':
        #     ligand_pos = ligand_pos + offset[batch_ligand]
        
        if pred_bond:

            return {
                'pos': ligand_pos,
                'v': ligand_v,
                'bond': pred_bonds,
                'pos_traj': pos_traj,
                'pos_cond_traj': pos_cond_traj,
                'pos_uncond_traj': pos_uncond_traj,
                'v_traj': v_traj,
                'v_cond_traj': v_cond_traj,
                'v_uncond_traj': v_uncond_traj,
                'v0_traj': v0_pred_traj,
                'vt_traj': vt_pred_traj
            }
        else:
            return {
                'pos': ligand_pos,
                'v': ligand_v,
                'bond': pred_bonds,
                'pos_traj': pos_traj,
                'pos_cond_traj': pos_cond_traj,
                'pos_uncond_traj': pos_uncond_traj,
                'v_traj': v_traj,
                'v_cond_traj': v_cond_traj,
                'v_uncond_traj': v_uncond_traj,
                'v0_traj': v0_pred_traj,
                'vt_traj': vt_pred_traj
            }
        
    @torch.no_grad()
    def sample_diffusion(self, init_ligand_pos, init_ligand_v, batch_ligand, pocket_data, ligand_shape=None, 
                         mask_shape_emb=None, num_steps=None, center_pos_mode=None, pred_bond=False):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_ligand.max().item() + 1
        # print('sample center pos mode: ', center_pos_mode)

        init_ligand_pos, offset = center_pos(init_ligand_pos, batch_ligand, mode=center_pos_mode)
        
        pos_traj, v_traj = [], []
        pos_cond_traj, v_cond_traj = [], []
        pos_uncond_traj, v_uncond_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        pred_bonds = None
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
        # for i in time_seq:
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)
            with torch.no_grad():
                preds_with_cond = self(
                    ligand_pos_perturbed=ligand_pos,
                    ligand_v_perturbed=ligand_v,
                    batch_ligand=batch_ligand,
                    pocket_data=pocket_data,
                    ligand_shape=ligand_shape,
                    mask_shape_emb=mask_shape_emb,
                    if_test=True,
                    pred_bond=True if self.pred_bond_type and pred_bond and i == 0 else False,
                    time_step=t
                )

            if self.pred_bond_type and pred_bond and i == 0:
                pred_bonds = torch.argmax(preds_with_cond['bond_pred'], dim=-1).view(-1, 1)
                pred_edge_indices = preds_with_cond['edge_index'].transpose(0, 1)
                pred_bonds = torch.cat([pred_edge_indices, pred_bonds], dim=1)

            preds = preds_with_cond

            pos_cond_traj.append(preds_with_cond['pred_ligand_pos'])
            v_cond_traj.append(preds_with_cond['pred_ligand_v'])
            
            if self.v_mode == 'tomask':
                #mask the mask atom
                preds['pred_ligand_v'][:, -1] = -1.e5

            # Compute posterior mean and variance
            with torch.no_grad():
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
                
                pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                # no noise when t == 0
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)

                ligand_pos = ligand_pos_next

                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes+int(self.v_mode=='tomask'))
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

                if center_pos_mode != 'none':
                    ori_ligand_pos = ligand_pos + offset[batch_ligand]
                else:
                    ori_ligand_pos = ligand_pos
                
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())
    
        # if center_pos_mode != 'none':
        #     ligand_pos = ligand_pos + offset[batch_ligand]
        
        if pred_bond:

            return {
                'pos': ligand_pos,
                'v': ligand_v,
                'bond': pred_bonds,
                'pos_traj': pos_traj,
                'pos_cond_traj': pos_cond_traj,
                'pos_uncond_traj': pos_uncond_traj,
                'v_traj': v_traj,
                'v_cond_traj': v_cond_traj,
                'v_uncond_traj': v_uncond_traj,
                'v0_traj': v0_pred_traj,
                'vt_traj': vt_pred_traj
            }
        else:
            return {
                'pos': ligand_pos,
                'v': ligand_v,
                'bond': pred_bonds,
                'pos_traj': pos_traj,
                'pos_cond_traj': pos_cond_traj,
                'pos_uncond_traj': pos_uncond_traj,
                'v_traj': v_traj,
                'v_cond_traj': v_cond_traj,
                'v_uncond_traj': v_uncond_traj,
                'v0_traj': v0_pred_traj,
                'vt_traj': vt_pred_traj
            }



    @torch.no_grad()
    def sample_diffusion_scaffold(
        self,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        pocket_data,
        fixed_mask,
        fixed_ligand_pos,
        fixed_ligand_v,
        ligand_shape=None,
        mask_shape_emb=None,
        num_steps=None,
        center_pos_mode=None,
        pred_bond=False,
        refine_steps=1,
        pca_perturb=None,
    ):
        if num_steps is None:
            num_steps = self.num_timesteps
        if refine_steps < 1:
            raise ValueError(f"refine_steps must be >= 1, got {refine_steps}")

        num_graphs = batch_ligand.max().item() + 1
        fixed_mask = torch.as_tensor(fixed_mask, device=batch_ligand.device).view(-1)
        if fixed_mask.numel() != batch_ligand.numel():
            raise ValueError(
                f"fixed_mask size mismatch: got {fixed_mask.numel()}, expected {batch_ligand.numel()}"
            )
        fixed_mask = fixed_mask == 0
        if init_ligand_v.dim() != 1:
            raise ValueError("sample_diffusion_scaffold currently expects categorical ligand_v (1D indices).")

        fixed_ligand_pos = torch.as_tensor(
            fixed_ligand_pos, device=init_ligand_pos.device, dtype=init_ligand_pos.dtype
        )
        fixed_ligand_v = torch.as_tensor(
            fixed_ligand_v, device=init_ligand_v.device, dtype=init_ligand_v.dtype
        )
        if fixed_ligand_pos.shape != init_ligand_pos.shape:
            raise ValueError(
                f"fixed_ligand_pos shape mismatch: got {fixed_ligand_pos.shape}, expected {init_ligand_pos.shape}"
            )
        if fixed_ligand_v.shape != init_ligand_v.shape:
            raise ValueError(
                f"fixed_ligand_v shape mismatch: got {fixed_ligand_v.shape}, expected {init_ligand_v.shape}"
            )

        init_ligand_pos, offset = center_pos(init_ligand_pos, batch_ligand, mode=center_pos_mode)
        if center_pos_mode != 'none':
            fixed_pos_centered = fixed_ligand_pos - offset[batch_ligand]
        else:
            fixed_pos_centered = fixed_ligand_pos
        pos_traj, v_traj = [], []
        pos_cond_traj, v_cond_traj = [], []
        pos_uncond_traj, v_uncond_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []

        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        ligand_pos[fixed_mask] = fixed_pos_centered[fixed_mask]
        ligand_v[fixed_mask] = fixed_ligand_v[fixed_mask]

        pred_bonds = None
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))

        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)

            x_t_cur = ligand_pos
            v_cur = ligand_v
            preds_with_cond = None
            log_ligand_v_recon = None
            log_model_prob = None

            for u in range(refine_steps):
                preds_with_cond = self(
                    ligand_pos_perturbed=x_t_cur,
                    ligand_v_perturbed=v_cur,
                    batch_ligand=batch_ligand,
                    pocket_data=pocket_data,
                    ligand_shape=ligand_shape,
                    mask_shape_emb=mask_shape_emb,
                    if_test=True,
                    pred_bond=True if self.pred_bond_type and pred_bond and i == 0 and u == (refine_steps - 1) else False,
                    time_step=t,
                    pca_perturb=pca_perturb,
                )

                preds = preds_with_cond
                if self.v_mode == 'tomask':
                    preds['pred_ligand_v'][:, -1] = -1.e5

                # Unknown part: reverse transition p_theta(x_{t-1} | x_t)
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']

                pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=x_t_cur, t=t, batch=batch_ligand)
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
                x_prev_unknown = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(x_t_cur)

                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_v_cur = index_to_log_onehot(v_cur, self.num_classes + int(self.v_mode == 'tomask'))
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_v_cur, t, batch_ligand)
                v_prev_unknown = log_sample_categorical(log_model_prob)

                # Known part: q(x_{t-1} | x0_known), q(v_{t-1} | v0_known)
                log_v0_known = index_to_log_onehot(
                    fixed_ligand_v, self.num_classes + int(self.v_mode == 'tomask')
                )
                if i > 0:
                    t_prev = torch.full(size=(num_graphs,), fill_value=i - 1, dtype=torch.long, device=ligand_pos.device)
                    a_prev = extract(self.alphas_cumprod, t_prev, batch_ligand)
                    x_prev_known = a_prev.sqrt() * fixed_pos_centered + (1.0 - a_prev).sqrt() * torch.randn_like(fixed_pos_centered)
                    log_v_prev_known = self.q_v_pred(log_v0_known, t_prev, batch_ligand)
                    v_prev_known = log_sample_categorical(log_v_prev_known)
                else:
                    x_prev_known = fixed_pos_centered
                    v_prev_known = fixed_ligand_v

                # Merge with mask
                x_prev = x_prev_unknown.clone()
                x_prev[fixed_mask] = x_prev_known[fixed_mask]
                v_prev = v_prev_unknown.clone()
                v_prev[fixed_mask] = v_prev_known[fixed_mask]

                # Re-noise q(x_t | x_{t-1}) for iterative refinement when u < U and t > 1
                if (u < refine_steps - 1) and (i > 1):
                    beta_t = extract(self.betas, t, batch_ligand)
                    x_t_cur = (1.0 - beta_t).sqrt() * x_prev + beta_t.sqrt() * torch.randn_like(x_prev)
                    log_v_prev = index_to_log_onehot(v_prev, self.num_classes + int(self.v_mode == 'tomask'))
                    log_v_t = self.q_v_pred_one_timestep(log_v_prev, t, batch_ligand)
                    v_cur = log_sample_categorical(log_v_t)
                else:
                    x_t_cur = x_prev
                    v_cur = v_prev

            if self.pred_bond_type and pred_bond and i == 0 and preds_with_cond is not None and 'bond_pred' in preds_with_cond:
                pred_bonds = torch.argmax(preds_with_cond['bond_pred'], dim=-1).view(-1, 1)
                pred_edge_indices = preds_with_cond['edge_index'].transpose(0, 1)
                pred_bonds = torch.cat([pred_edge_indices, pred_bonds], dim=1)

            ligand_pos = x_t_cur
            ligand_v = v_cur

            if preds_with_cond is not None:
                pos_cond_traj.append(preds_with_cond['pred_ligand_pos'])
                v_cond_traj.append(preds_with_cond['pred_ligand_v'])
            if log_ligand_v_recon is not None:
                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
            if log_model_prob is not None:
                vt_pred_traj.append(log_model_prob.clone().cpu())

            if center_pos_mode != 'none':
                ori_ligand_pos = ligand_pos + offset[batch_ligand]
            else:
                ori_ligand_pos = ligand_pos

            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        if center_pos_mode != 'none':
            ligand_pos = ligand_pos + offset[batch_ligand]

        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'bond': pred_bonds,
            'pos_traj': pos_traj,
            'pos_cond_traj': pos_cond_traj,
            'pos_uncond_traj': pos_uncond_traj,
            'v_traj': v_traj,
            'v_cond_traj': v_cond_traj,
            'v_uncond_traj': v_uncond_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj
        }

    @torch.no_grad()
    def sample_scaffold(
        self,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        pocket_data,
        scaffold_mask,
        fixed_ligand_pos,
        fixed_ligand_v,
        ligand_shape=None,
        mask_shape_emb=None,
        num_steps=None,
        center_pos_mode=None,
        pred_bond=False,
    ):
        """
        Sampling aligned with scaffold-training: unscaffold is fixed/clean, scaffold is generated/noisy.
        The scaffold mask follows the dataset convention where scaffold_mask == 1 indicates scaffold atoms.
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        num_graphs = batch_ligand.max().item() + 1
        scaffold_mask = torch.as_tensor(scaffold_mask, device=batch_ligand.device).view(-1)
        if scaffold_mask.numel() != batch_ligand.numel():
            raise ValueError(
                f"scaffold_mask size mismatch: got {scaffold_mask.numel()}, expected {batch_ligand.numel()}"
            )

        generated_mask = scaffold_mask == 1
        fixed_mask = ~generated_mask

        if init_ligand_v.dim() != 1:
            raise ValueError("sample_scaffold currently expects categorical ligand_v (1D indices).")

        fixed_ligand_pos = torch.as_tensor(
            fixed_ligand_pos, device=init_ligand_pos.device, dtype=init_ligand_pos.dtype
        )
        fixed_ligand_v = torch.as_tensor(
            fixed_ligand_v, device=init_ligand_v.device, dtype=init_ligand_v.dtype
        )
        if fixed_ligand_pos.shape != init_ligand_pos.shape:
            raise ValueError(
                f"fixed_ligand_pos shape mismatch: got {fixed_ligand_pos.shape}, expected {init_ligand_pos.shape}"
            )
        if fixed_ligand_v.shape != init_ligand_v.shape:
            raise ValueError(
                f"fixed_ligand_v shape mismatch: got {fixed_ligand_v.shape}, expected {init_ligand_v.shape}"
            )

        init_ligand_pos, offset = center_pos(init_ligand_pos, batch_ligand, mode=center_pos_mode)
        if center_pos_mode != 'none':
            fixed_pos_centered = fixed_ligand_pos - offset[batch_ligand]
        else:
            fixed_pos_centered = fixed_ligand_pos

        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        ligand_pos[fixed_mask] = fixed_pos_centered[fixed_mask]
        ligand_v[fixed_mask] = fixed_ligand_v[fixed_mask]

        pos_traj, v_traj = [], []
        pos_cond_traj, v_cond_traj = [], []
        pos_uncond_traj, v_uncond_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        pred_bonds = None

        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)
            with torch.no_grad():
                preds_with_cond = self(
                    ligand_pos_perturbed=ligand_pos,
                    ligand_v_perturbed=ligand_v,
                    batch_ligand=batch_ligand,
                    pocket_data=pocket_data,
                    ligand_shape=ligand_shape,
                    mask_shape_emb=mask_shape_emb,
                    if_test=True,
                    pred_bond=True if self.pred_bond_type and pred_bond and i == 0 else False,
                    time_step=t,
                )

            if self.pred_bond_type and pred_bond and i == 0:
                pred_bonds = torch.argmax(preds_with_cond['bond_pred'], dim=-1).view(-1, 1)
                pred_edge_indices = preds_with_cond['edge_index'].transpose(0, 1)
                pred_bonds = torch.cat([pred_edge_indices, pred_bonds], dim=1)

            preds = preds_with_cond
            pos_cond_traj.append(preds_with_cond['pred_ligand_pos'])
            v_cond_traj.append(preds_with_cond['pred_ligand_v'])
            if self.v_mode == 'tomask':
                preds['pred_ligand_v'][:, -1] = -1.e5

            pos0_from_e = preds['pred_ligand_pos']
            v0_from_e = preds['pred_ligand_v']

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            x_prev = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)

            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes + int(self.v_mode == 'tomask'))
            log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
            v_prev = log_sample_categorical(log_model_prob)

            # Keep fixed (unscaffold) part clean throughout sampling.
            x_prev[fixed_mask] = fixed_pos_centered[fixed_mask]
            v_prev[fixed_mask] = fixed_ligand_v[fixed_mask]

            ligand_pos = x_prev
            ligand_v = v_prev

            v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
            vt_pred_traj.append(log_model_prob.clone().cpu())

            if center_pos_mode != 'none':
                ori_ligand_pos = ligand_pos + offset[batch_ligand]
            else:
                ori_ligand_pos = ligand_pos

            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        if center_pos_mode != 'none':
            ligand_pos = ligand_pos + offset[batch_ligand]

        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'bond': pred_bonds,
            'pos_traj': pos_traj,
            'pos_cond_traj': pos_cond_traj,
            'pos_uncond_traj': pos_uncond_traj,
            'v_traj': v_traj,
            'v_cond_traj': v_cond_traj,
            'v_uncond_traj': v_uncond_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj
        }

    def _log_q_generated_xt_given_x0_single(
        self,
        xt_pos,
        xt_v,
        x0_pos,
        x0_v,
        generated_mask,
        t_scalar,
    ):
        generated_mask = generated_mask.bool()
        if generated_mask.numel() != xt_pos.size(0):
            raise ValueError(
                f"generated_mask size mismatch: got {generated_mask.numel()}, expected {xt_pos.size(0)}"
            )
        if generated_mask.sum().item() == 0:
            raise ValueError("generated part is empty when estimating q(x_t | x_0).")

        xt_pos_gen = xt_pos[generated_mask]
        x0_pos_gen = x0_pos[generated_mask]
        xt_v_gen = xt_v[generated_mask]
        x0_v_gen = x0_v[generated_mask]

        alpha_bar_t = self.alphas_cumprod[t_scalar].to(device=xt_pos.device, dtype=xt_pos.dtype)
        pos_mean = alpha_bar_t.sqrt() * x0_pos_gen
        pos_var = (1.0 - alpha_bar_t).clamp_min(1e-30)
        pos_diff = xt_pos_gen - pos_mean
        pos_log_scale = 0.5 * torch.log(pos_var)
        pos_log_scale = torch.full_like(xt_pos_gen, pos_log_scale)
        log_q_pos = log_normal(xt_pos_gen, pos_mean, pos_log_scale).sum()

        num_classes = self.num_classes + int(self.v_mode == 'tomask')
        log_x0_v = index_to_log_onehot(x0_v_gen, num_classes)
        batch_single = torch.zeros(x0_v_gen.size(0), dtype=torch.long, device=xt_v.device)
        t_batch = torch.full((1,), int(t_scalar), dtype=torch.long, device=xt_v.device)
        log_q_v = self.q_v_pred(log_x0_v, t_batch, batch_single)
        log_xt_v = index_to_log_onehot(xt_v_gen, num_classes)
        log_q_v_val = log_categorical(log_xt_v, log_q_v).sum()
        return log_q_pos + log_q_v_val

    def _estimate_log_p_xref_given_xt_generated(
        self,
        xt_pos,
        xt_v,
        xref_pos,
        xref_v,
        mc_x0_pos,
        mc_x0_v,
        anchor_pos,
        generated_mask,
        batch_ligand,
        time_step,
        pos_weight=1.0,
        type_weight=1.0,
        family_x0_pos=None,
        family_x0_v=None,
        family_reduce='max',
        return_components=False,
    ):
        num_graphs = batch_ligand.max().item() + 1
        if xt_pos.size(0) % num_graphs != 0:
            raise ValueError(
                f"xt_pos size {xt_pos.size(0)} is not divisible by num_graphs {num_graphs}"
            )

        num_atoms = xt_pos.size(0) // num_graphs
        if generated_mask.numel() != xt_pos.size(0):
            raise ValueError(
                f"generated_mask size mismatch: got {generated_mask.numel()}, expected {xt_pos.size(0)}"
            )
        if mc_x0_pos.size(1) != num_atoms:
            raise ValueError(
                f"mc_x0_pos atom count mismatch: got {mc_x0_pos.size(1)}, expected {num_atoms}"
            )
        if mc_x0_v.size(1) != num_atoms:
            raise ValueError(
                f"mc_x0_v atom count mismatch: got {mc_x0_v.size(1)}, expected {num_atoms}"
            )

        alignment = self._build_generated_alignment(
            xref_pos=xref_pos,
            xref_v=xref_v,
            mc_x0_pos=mc_x0_pos,
            mc_x0_v=mc_x0_v,
            anchor_pos=anchor_pos,
            generated_mask=generated_mask,
            num_graphs=num_graphs,
            num_atoms=num_atoms,
            device=xt_pos.device,
            family_x0_pos=family_x0_pos,
            family_x0_v=family_x0_v,
        )

        return self._estimate_log_p_xref_given_xt_generated_from_alignment(
            xt_pos=xt_pos,
            xt_v=xt_v,
            time_step=time_step,
            alignment=alignment,
            pos_weight=pos_weight,
            type_weight=type_weight,
            family_reduce=family_reduce,
            return_components=return_components,
        )

    def _build_generated_alignment(
        self,
        xref_pos,
        xref_v,
        mc_x0_pos,
        mc_x0_v,
        anchor_pos,
        generated_mask,
        num_graphs,
        num_atoms,
        device,
        family_x0_pos=None,
        family_x0_v=None,
    ):
        xref_pos_batched = xref_pos.reshape(num_graphs, num_atoms, 3)
        xref_v_batched = xref_v.reshape(num_graphs, num_atoms)
        anchor_pos_batched = anchor_pos.reshape(num_graphs, num_atoms, 3)
        generated_mask_batched = generated_mask.reshape(num_graphs, num_atoms).bool()

        if not torch.all(generated_mask_batched == generated_mask_batched[0:1]):
            raise ValueError("Batch version currently expects the same generated mask template for all graphs.")
        generated_mask_single = generated_mask_batched[0]
        num_generated = int(generated_mask_single.sum().item())
        if num_generated == 0:
            raise ValueError("generated part is empty when estimating p(x_ref | x_t).")

        xref_pos_gen = xref_pos_batched[:, generated_mask_single, :]
        xref_v_gen = xref_v_batched[:, generated_mask_single]
        mc_x0_pos_gen = mc_x0_pos[:, generated_mask_single, :]
        mc_x0_v_gen = mc_x0_v[:, generated_mask_single]
        anchor_pos_gen = anchor_pos_batched[:, generated_mask_single, :]

        alignment = {
            'num_graphs': num_graphs,
            'num_atoms': num_atoms,
            'num_generated': num_generated,
            'generated_mask_single': generated_mask_single,
            'aligned_xref_pos_gen': None,
            'aligned_xref_v_gen': None,
            'aligned_mc_x0_pos_gen': None,
            'aligned_mc_x0_v_gen': None,
            'aligned_family_x0_pos_gen': None,
            'aligned_family_x0_v_gen': None,
        }

        aligned_xref_pos_gen = torch.empty_like(xref_pos_gen)
        aligned_xref_v_gen = torch.empty_like(xref_v_gen)
        aligned_mc_x0_pos_gen = torch.empty_like(mc_x0_pos_gen.unsqueeze(0).expand(num_graphs, -1, -1, -1))
        aligned_mc_x0_v_gen = torch.empty_like(mc_x0_v_gen.unsqueeze(0).expand(num_graphs, -1, -1))

        for graph_idx in range(num_graphs):
            anchor_graph = anchor_pos_gen[graph_idx]
            ref_cost = torch.cdist(anchor_graph, xref_pos_gen[graph_idx], p=2).pow(2).detach().cpu().numpy()
            _, ref_col_ind = linear_sum_assignment(ref_cost)
            ref_perm = torch.as_tensor(ref_col_ind, device=device, dtype=torch.long)
            aligned_xref_pos_gen[graph_idx] = xref_pos_gen[graph_idx][ref_perm]
            aligned_xref_v_gen[graph_idx] = xref_v_gen[graph_idx][ref_perm]

            for mc_idx in range(mc_x0_pos_gen.size(0)):
                mc_cost = torch.cdist(anchor_graph, mc_x0_pos_gen[mc_idx], p=2).pow(2).detach().cpu().numpy()
                _, mc_col_ind = linear_sum_assignment(mc_cost)
                mc_perm = torch.as_tensor(mc_col_ind, device=device, dtype=torch.long)
                aligned_mc_x0_pos_gen[graph_idx, mc_idx] = mc_x0_pos_gen[mc_idx][mc_perm]
                aligned_mc_x0_v_gen[graph_idx, mc_idx] = mc_x0_v_gen[mc_idx][mc_perm]

        alignment['aligned_xref_pos_gen'] = aligned_xref_pos_gen
        alignment['aligned_xref_v_gen'] = aligned_xref_v_gen
        alignment['aligned_mc_x0_pos_gen'] = aligned_mc_x0_pos_gen
        alignment['aligned_mc_x0_v_gen'] = aligned_mc_x0_v_gen

        if family_x0_pos is not None and family_x0_v is not None:
            if family_x0_pos.dim() != 3 or family_x0_v.dim() != 2:
                raise ValueError("family_x0_pos/family_x0_v must have shapes [K, N, 3] and [K, N].")
            if family_x0_pos.size(1) != num_atoms or family_x0_v.size(1) != num_atoms:
                raise ValueError(
                    f"family atom count mismatch: got {family_x0_pos.size(1)} / {family_x0_v.size(1)}, expected {num_atoms}"
                )

            family_x0_pos_gen = family_x0_pos[:, generated_mask_single, :]
            family_x0_v_gen = family_x0_v[:, generated_mask_single]
            num_family = family_x0_pos_gen.size(0)

            aligned_family_x0_pos_gen = torch.empty_like(
                family_x0_pos_gen.unsqueeze(0).expand(num_graphs, -1, -1, -1)
            )
            aligned_family_x0_v_gen = torch.empty_like(
                family_x0_v_gen.unsqueeze(0).expand(num_graphs, -1, -1)
            )

            for graph_idx in range(num_graphs):
                anchor_graph = anchor_pos_gen[graph_idx]
                for family_idx in range(num_family):
                    fam_cost = torch.cdist(anchor_graph, family_x0_pos_gen[family_idx], p=2).pow(2).detach().cpu().numpy()
                    _, fam_col_ind = linear_sum_assignment(fam_cost)
                    fam_perm = torch.as_tensor(fam_col_ind, device=device, dtype=torch.long)
                    aligned_family_x0_pos_gen[graph_idx, family_idx] = family_x0_pos_gen[family_idx][fam_perm]
                    aligned_family_x0_v_gen[graph_idx, family_idx] = family_x0_v_gen[family_idx][fam_perm]

            alignment['aligned_family_x0_pos_gen'] = aligned_family_x0_pos_gen
            alignment['aligned_family_x0_v_gen'] = aligned_family_x0_v_gen

        return alignment

    def _estimate_log_p_xref_given_xt_generated_from_alignment(
        self,
        xt_pos,
        xt_v,
        time_step,
        alignment,
        pos_weight=1.0,
        type_weight=1.0,
        family_reduce='max',
        return_components=False,
    ):
        if xt_pos.dim() == 2:
            xt_pos = xt_pos.unsqueeze(0)
        elif xt_pos.dim() != 3:
            raise ValueError(f"xt_pos must have shape [BN,3] or [C,BN,3], got {tuple(xt_pos.shape)}")
        if xt_v.dim() == 1:
            xt_v = xt_v.unsqueeze(0)
        elif xt_v.dim() != 2:
            raise ValueError(f"xt_v must have shape [BN] or [C,BN], got {tuple(xt_v.shape)}")
        if xt_pos.size(0) != xt_v.size(0):
            raise ValueError("xt_pos/xt_v candidate dimension mismatch.")

        num_candidates = xt_pos.size(0)
        num_graphs = alignment['num_graphs']
        num_atoms = alignment['num_atoms']
        num_generated = alignment['num_generated']
        generated_mask_single = alignment['generated_mask_single']
        aligned_xref_pos_gen = alignment['aligned_xref_pos_gen']
        aligned_xref_v_gen = alignment['aligned_xref_v_gen']
        aligned_mc_x0_pos_gen = alignment['aligned_mc_x0_pos_gen']
        aligned_mc_x0_v_gen = alignment['aligned_mc_x0_v_gen']
        aligned_family_x0_pos_gen = alignment['aligned_family_x0_pos_gen']
        aligned_family_x0_v_gen = alignment['aligned_family_x0_v_gen']

        if xt_pos.size(1) != num_graphs * num_atoms or xt_v.size(1) != num_graphs * num_atoms:
            raise ValueError(
                f"xt candidate size mismatch: expected {num_graphs * num_atoms}, "
                f"got {xt_pos.size(1)} / {xt_v.size(1)}"
            )
        if time_step.numel() != num_graphs:
            raise ValueError(f"time_step must have length {num_graphs}, got {time_step.numel()}")

        xt_pos_batched = xt_pos.reshape(num_candidates, num_graphs, num_atoms, 3)
        xt_v_batched = xt_v.reshape(num_candidates, num_graphs, num_atoms)
        xt_pos_gen = xt_pos_batched[:, :, generated_mask_single, :]
        xt_v_gen = xt_v_batched[:, :, generated_mask_single]

        alpha_bar_t = self.alphas_cumprod.index_select(0, time_step).to(device=xt_pos.device, dtype=xt_pos.dtype)
        pos_var = (1.0 - alpha_bar_t).clamp_min(1e-30)
        pos_log_scale = 0.5 * torch.log(pos_var)
        pos_const = -0.5 * np.log(2 * np.pi)

        ref_pos_mean = alpha_bar_t[:, None, None].sqrt() * aligned_xref_pos_gen
        ref_pos_diff = xt_pos_gen - ref_pos_mean.unsqueeze(0)
        # ref_pos_log_q = (
        #     -0.5 * ref_pos_diff.pow(2) / pos_var[:, None, None]
        #     - pos_log_scale[:, None, None]
        #     + pos_const
        # ).sum(dim=(1, 2))

        ref_pos_log_q = (-0.5 * ref_pos_diff.pow(2)).sum(dim=(2, 3)) / float(num_generated)

        num_classes = self.num_classes + int(self.v_mode == 'tomask')
        log_xt_v = index_to_log_onehot(
            xt_v_gen.reshape(-1), num_classes
        ).reshape(num_candidates, num_graphs, num_generated, num_classes)

        ref_log_x0_v = index_to_log_onehot(aligned_xref_v_gen.reshape(-1), num_classes)
        ref_batch = torch.arange(num_graphs, device=xt_v.device).repeat_interleave(num_generated)
        ref_log_q_v = self.q_v_pred(ref_log_x0_v, time_step, ref_batch).reshape(num_graphs, num_generated, num_classes)
        ref_log_q_v_val = (log_xt_v.exp() * ref_log_q_v.unsqueeze(0)).sum(dim=-1).sum(dim=-1) / float(num_generated)
        log_q_ref = pos_weight * ref_pos_log_q + type_weight * ref_log_q_v_val

        num_mc = aligned_mc_x0_pos_gen.size(1)
        mc_pos_mean = alpha_bar_t[:, None, None, None].sqrt() * aligned_mc_x0_pos_gen
        mc_pos_diff = xt_pos_gen[:, :, None, :, :] - mc_pos_mean.unsqueeze(0)
        # mc_pos_log_q = (
        #     -0.5 * mc_pos_diff.pow(2) / pos_var[:, None, None, None]
        #     - pos_log_scale[:, None, None, None]
        #     + pos_const
        # ).sum(dim=(2, 3))

        mc_pos_log_q = (-0.5 * mc_pos_diff.pow(2)).sum(dim=(3, 4)) / float(num_generated)

        mc_log_x0_v = index_to_log_onehot(aligned_mc_x0_v_gen.reshape(-1), num_classes)
        mc_time_step = time_step[:, None].expand(num_graphs, num_mc).reshape(-1)
        mc_time_step = mc_time_step.repeat_interleave(num_generated)
        mc_batch = torch.arange(num_graphs * num_mc, device=xt_v.device).repeat_interleave(num_generated)
        mc_log_q_v = self.q_v_pred(mc_log_x0_v, mc_time_step, mc_batch).reshape(num_graphs, num_mc, num_generated, num_classes)

        log_xt_v_mc = log_xt_v[:, :, None, :, :].expand(num_candidates, num_graphs, num_mc, num_generated, num_classes)
        mc_log_q_v_val = (log_xt_v_mc.exp() * mc_log_q_v.unsqueeze(0)).sum(dim=-1).sum(dim=-1) / float(num_generated)
        mc_log_q = pos_weight * mc_pos_log_q + type_weight * mc_log_q_v_val

        family_debug = {}
        if aligned_family_x0_pos_gen is not None and aligned_family_x0_v_gen is not None:
            num_family = aligned_family_x0_pos_gen.size(1)
            family_pos_mean = alpha_bar_t[:, None, None, None].sqrt() * aligned_family_x0_pos_gen
            family_pos_diff = xt_pos_gen[:, :, None, :, :] - family_pos_mean.unsqueeze(0)
            family_pos_log_q = (-0.5 * family_pos_diff.pow(2)).sum(dim=(3, 4)) / float(num_generated)

            family_log_x0_v = index_to_log_onehot(aligned_family_x0_v_gen.reshape(-1), num_classes)
            family_time_step = time_step[:, None].expand(num_graphs, num_family).reshape(-1)
            family_time_step = family_time_step.repeat_interleave(num_generated)
            family_batch = torch.arange(num_graphs * num_family, device=xt_v.device).repeat_interleave(num_generated)
            family_log_q_v = self.q_v_pred(
                family_log_x0_v, family_time_step, family_batch
            ).reshape(num_graphs, num_family, num_generated, num_classes)
            log_xt_v_family = log_xt_v[:, :, None, :, :].expand(num_candidates, num_graphs, num_family, num_generated, num_classes)
            family_log_q_v_val = (log_xt_v_family.exp() * family_log_q_v.unsqueeze(0)).sum(dim=-1).sum(dim=-1) / float(num_generated)
            family_log_q = pos_weight * family_pos_log_q + type_weight * family_log_q_v_val

            if family_reduce == 'max':
                family_log_q_reduced, family_argmax = family_log_q.max(dim=2)
                ref_pos_log_q = family_pos_log_q.gather(2, family_argmax.unsqueeze(-1)).squeeze(-1)
                ref_log_q_v_val = family_log_q_v_val.gather(2, family_argmax.unsqueeze(-1)).squeeze(-1)
            elif family_reduce == 'logsumexp':
                log_num_family = float(np.log(num_family))
                family_log_q_reduced = torch.logsumexp(family_log_q, dim=2) - log_num_family
                ref_pos_log_q = torch.logsumexp(family_pos_log_q, dim=2) - log_num_family
                ref_log_q_v_val = torch.logsumexp(family_log_q_v_val, dim=2) - log_num_family
            else:
                raise ValueError(f"Unsupported family_reduce: {family_reduce}")

            log_q_ref = family_log_q_reduced
            family_debug = {
                'family_size': num_family,
                'family_log_q_mean': family_log_q.mean(dim=2),
            }

        log_p_xt = torch.logsumexp(mc_log_q, dim=2) - np.log(num_mc)
        log_post = log_q_ref - log_p_xt

        if num_candidates == 1:
            log_q_ref = log_q_ref.squeeze(0)
            log_p_xt = log_p_xt.squeeze(0)
            log_post = log_post.squeeze(0)
            ref_pos_log_q = ref_pos_log_q.squeeze(0)
            ref_log_q_v_val = ref_log_q_v_val.squeeze(0)
            mc_pos_log_q_mean = mc_pos_log_q.mean(dim=2).squeeze(0)
            mc_log_q_v_val_mean = mc_log_q_v_val.mean(dim=2).squeeze(0)
            mc_log_q_mean = mc_log_q.mean(dim=2).squeeze(0)
            if family_debug:
                family_debug = {
                    'family_size': family_debug['family_size'],
                    'family_log_q_mean': family_debug['family_log_q_mean'].squeeze(0),
                }
        else:
            mc_pos_log_q_mean = mc_pos_log_q.mean(dim=2)
            mc_log_q_v_val_mean = mc_log_q_v_val.mean(dim=2)
            mc_log_q_mean = mc_log_q.mean(dim=2)

        if not return_components:
            return log_q_ref, log_p_xt, log_post

        return {
            'log_q_ref': log_q_ref,
            'log_p_xt': log_p_xt,
            'log_post': log_post,
            'ref_pos_log_q': ref_pos_log_q,
            'ref_type_log_q': ref_log_q_v_val,
            'mc_pos_log_q_mean': mc_pos_log_q_mean,
            'mc_type_log_q_mean': mc_log_q_v_val_mean,
            'mc_log_q_mean': mc_log_q_mean,
            **family_debug,
        }

    @torch.no_grad()
    def sample_scaffold_guided(
        self,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        pocket_data,
        scaffold_mask,
        fixed_ligand_pos,
        fixed_ligand_v,
        xref_ligand_pos,
        xref_ligand_v,
        mc_x0_pos,
        mc_x0_v,
        family_x0_pos=None,
        family_x0_v=None,
        positive_family_x0_pos=None,
        positive_family_x0_v=None,
        family_reduce='max',
        use_complete_dissimilar_search=False,
        search_positive_score_max=2.0,
        guidance_eval_steps=None,
        search_steps=None,
        search_num_candidates=5,
        search_pos_noise_scale=1.0,
        search_type_temperature=1.0,
        pos_weight=1.0,
        type_weight=1.0,
        debug_logp=False,
        ligand_shape=None,
        mask_shape_emb=None,
        num_steps=None,
        center_pos_mode=None,
        pred_bond=False,
        store_trajectories=False,
    ):
        """
        Guided scaffold sampling that records log q(x_t^gen | x_ref^gen), Monte Carlo
        log p(x_t^gen), and the resulting log p(x_ref^gen | x_t^gen) up to a constant.
        This first version only estimates scores and does not alter the denoising trajectory.
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        num_graphs = batch_ligand.max().item() + 1
        scaffold_mask = torch.as_tensor(scaffold_mask, device=batch_ligand.device).view(-1)
        if scaffold_mask.numel() != batch_ligand.numel():
            raise ValueError(
                f"scaffold_mask size mismatch: got {scaffold_mask.numel()}, expected {batch_ligand.numel()}"
            )

        generated_mask = scaffold_mask == 1
        fixed_mask = ~generated_mask

        if init_ligand_v.dim() != 1:
            raise ValueError("sample_scaffold_guided currently expects categorical ligand_v (1D indices).")

        fixed_ligand_pos = torch.as_tensor(
            fixed_ligand_pos, device=init_ligand_pos.device, dtype=init_ligand_pos.dtype
        )
        fixed_ligand_v = torch.as_tensor(
            fixed_ligand_v, device=init_ligand_v.device, dtype=init_ligand_v.dtype
        )
        xref_ligand_pos = torch.as_tensor(
            xref_ligand_pos, device=init_ligand_pos.device, dtype=init_ligand_pos.dtype
        )
        xref_ligand_v = torch.as_tensor(
            xref_ligand_v, device=init_ligand_v.device, dtype=init_ligand_v.dtype
        )
        mc_x0_pos = torch.as_tensor(
            mc_x0_pos, device=init_ligand_pos.device, dtype=init_ligand_pos.dtype
        )
        mc_x0_v = torch.as_tensor(
            mc_x0_v, device=init_ligand_v.device, dtype=init_ligand_v.dtype
        )

        if fixed_ligand_pos.shape != init_ligand_pos.shape:
            raise ValueError(
                f"fixed_ligand_pos shape mismatch: got {fixed_ligand_pos.shape}, expected {init_ligand_pos.shape}"
            )
        if fixed_ligand_v.shape != init_ligand_v.shape:
            raise ValueError(
                f"fixed_ligand_v shape mismatch: got {fixed_ligand_v.shape}, expected {init_ligand_v.shape}"
            )
        if xref_ligand_pos.shape != init_ligand_pos.shape:
            raise ValueError(
                f"xref_ligand_pos shape mismatch: got {xref_ligand_pos.shape}, expected {init_ligand_pos.shape}"
            )
        if xref_ligand_v.shape != init_ligand_v.shape:
            raise ValueError(
                f"xref_ligand_v shape mismatch: got {xref_ligand_v.shape}, expected {init_ligand_v.shape}"
            )
        if mc_x0_pos.dim() != 3:
            raise ValueError(f"mc_x0_pos must have shape [M, N, 3], got {tuple(mc_x0_pos.shape)}")
        if mc_x0_v.dim() != 2:
            raise ValueError(f"mc_x0_v must have shape [M, N], got {tuple(mc_x0_v.shape)}")

        init_ligand_pos, offset = center_pos(init_ligand_pos, batch_ligand, mode=center_pos_mode)
        if center_pos_mode != 'none':
            fixed_pos_centered = fixed_ligand_pos - offset[batch_ligand]
            xref_pos_centered = xref_ligand_pos - offset[batch_ligand]
        else:
            fixed_pos_centered = fixed_ligand_pos
            xref_pos_centered = xref_ligand_pos

        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        ligand_pos[fixed_mask] = fixed_pos_centered[fixed_mask]
        ligand_v[fixed_mask] = fixed_ligand_v[fixed_mask]

        pos_traj, v_traj = [], []
        pos_cond_traj, v_cond_traj = [], []
        pos_uncond_traj, v_uncond_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        log_q_xt_given_xref_traj = []
        log_p_xt_mc_traj = []
        log_p_xref_given_xt_traj = []
        score_time_steps = []
        pred_bonds = None

        eval_step_set = None
        if guidance_eval_steps is not None:
            eval_step_set = {int(step) for step in guidance_eval_steps}
        search_step_set = None
        if search_steps is not None:
            search_step_set = {int(step) for step in search_steps}

        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)

            preds_with_cond = self(
                ligand_pos_perturbed=ligand_pos,
                ligand_v_perturbed=ligand_v,
                batch_ligand=batch_ligand,
                pocket_data=pocket_data,
                ligand_shape=ligand_shape,
                mask_shape_emb=mask_shape_emb,
                if_test=True,
                pred_bond=True if self.pred_bond_type and pred_bond and i == 0 else False,
                time_step=t,
            )

            if self.pred_bond_type and pred_bond and i == 0:
                pred_bonds = torch.argmax(preds_with_cond['bond_pred'], dim=-1).view(-1, 1)
                pred_edge_indices = preds_with_cond['edge_index'].transpose(0, 1)
                pred_bonds = torch.cat([pred_edge_indices, pred_bonds], dim=1)

            preds = preds_with_cond
            if store_trajectories:
                pos_cond_traj.append(preds_with_cond['pred_ligand_pos'])
                v_cond_traj.append(preds_with_cond['pred_ligand_v'])

            if self.v_mode == 'tomask':
                preds['pred_ligand_v'][:, -1] = -1.e5

            pos0_from_e = preds['pred_ligand_pos']
            v0_from_e = preds['pred_ligand_v']

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            x_prev = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)

            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes + int(self.v_mode == 'tomask'))
            log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
            v_prev = log_sample_categorical(log_model_prob)

            x_prev[fixed_mask] = fixed_pos_centered[fixed_mask]
            v_prev[fixed_mask] = fixed_ligand_v[fixed_mask]

            next_t_scalar = i - 1
            # score_t_scalar = next_t_scalar if next_t_scalar >= 0 else 0
            score_t = torch.full(size=(num_graphs,), fill_value=next_t_scalar, dtype=torch.long, device=ligand_pos.device)
            use_search = (
                search_step_set is not None
                and next_t_scalar >= 0
                and next_t_scalar in search_step_set
                and search_num_candidates > 1
            )
            use_evaluate = (
                eval_step_set is not None
                and next_t_scalar >= 0
                and next_t_scalar in eval_step_set
            )

            score_alignment = None
            positive_score_alignment = None
            if use_search or use_evaluate and not use_complete_dissimilar_search:
                num_atoms = ligand_pos.size(0) // num_graphs
                score_alignment = self._build_generated_alignment(
                    xref_pos=xref_pos_centered,
                    xref_v=xref_ligand_v,
                    mc_x0_pos=mc_x0_pos,
                    mc_x0_v=mc_x0_v,
                    anchor_pos=pos0_from_e,
                    generated_mask=generated_mask,
                    num_graphs=num_graphs,
                    num_atoms=num_atoms,
                    device=ligand_pos.device,
                    family_x0_pos=family_x0_pos,
                    family_x0_v=family_x0_v,
                )
            if use_search and use_complete_dissimilar_search and positive_family_x0_pos is not None and positive_family_x0_v is not None:
                positive_score_alignment = self._build_generated_alignment(
                    xref_pos=xref_pos_centered,
                    xref_v=xref_ligand_v,
                    mc_x0_pos=mc_x0_pos,
                    mc_x0_v=mc_x0_v,
                    anchor_pos=pos0_from_e,
                    generated_mask=generated_mask,
                    num_graphs=num_graphs,
                    num_atoms=num_atoms,
                    device=ligand_pos.device,
                    family_x0_pos=positive_family_x0_pos,
                    family_x0_v=positive_family_x0_v,
                )

            if use_search:
                num_atoms = ligand_pos.size(0) // num_graphs
                std = nonzero_mask * (0.5 * pos_log_variance).exp() * float(search_pos_noise_scale)
                if search_type_temperature <= 0:
                    raise ValueError("search_type_temperature must be positive.")
                search_log_model_prob = log_model_prob / float(search_type_temperature)
                original_log_post = self._estimate_log_p_xref_given_xt_generated_from_alignment(
                    xt_pos=x_prev,
                    xt_v=v_prev,
                    time_step=score_t,
                    alignment=score_alignment,
                    pos_weight=pos_weight,
                    type_weight=type_weight,
                    family_reduce='max',
                    return_components=False,
                )[2]
                pos_candidates = []
                v_candidates = []

                for _ in range(search_num_candidates):
                    cand_pos = pos_model_mean + std * torch.randn_like(ligand_pos)
                    cand_v = log_sample_categorical(search_log_model_prob)
                    cand_pos[fixed_mask] = fixed_pos_centered[fixed_mask]
                    cand_v[fixed_mask] = fixed_ligand_v[fixed_mask]
                    pos_candidates.append(cand_pos)
                    v_candidates.append(cand_v)

                pos_candidates = torch.stack(pos_candidates, dim=0).reshape(search_num_candidates, num_graphs, num_atoms, 3)
                v_candidates = torch.stack(v_candidates, dim=0).reshape(search_num_candidates, num_graphs, num_atoms)
                xt_candidates_flat_pos = pos_candidates.reshape(search_num_candidates, num_graphs * num_atoms, 3)
                xt_candidates_flat_v = v_candidates.reshape(search_num_candidates, num_graphs * num_atoms)

                if use_complete_dissimilar_search:
                    original_positive_log_post = self._estimate_log_p_xref_given_xt_generated_from_alignment(
                        xt_pos=x_prev,
                        xt_v=v_prev,
                        time_step=score_t,
                        alignment=positive_score_alignment,
                        pos_weight=pos_weight,
                        type_weight=type_weight,
                        family_reduce='max',
                        return_components=False,
                    )[2]
                    _, _, positive_score_candidates = self._estimate_log_p_xref_given_xt_generated_from_alignment(
                        xt_pos=xt_candidates_flat_pos,
                        xt_v=xt_candidates_flat_v,
                        time_step=score_t,
                        alignment=positive_score_alignment,
                        pos_weight=pos_weight,
                        type_weight=type_weight,
                        family_reduce='max',
                        return_components=False,
                    )
                    valid_candidate_mask = positive_score_candidates <= float(search_positive_score_max)
                    masked_scores = torch.where(
                        valid_candidate_mask,
                        positive_score_candidates,
                        torch.full_like(positive_score_candidates, float('-inf')),
                    )
                    best_idx = torch.argmax(masked_scores, dim=0)
                    has_valid_candidate = valid_candidate_mask.any(dim=0)
                    debug_selected_scores = torch.where(
                        has_valid_candidate,
                        positive_score_candidates[best_idx, torch.arange(num_graphs, device=ligand_pos.device)],
                        original_positive_log_post,
                    )
                    debug_original_scores = original_positive_log_post
                    debug_valid_candidate_mask = valid_candidate_mask
                    debug_score_label = "positive_log_post"
                else:
                    _, _, score_candidates = self._estimate_log_p_xref_given_xt_generated_from_alignment(
                        xt_pos=xt_candidates_flat_pos,
                        xt_v=xt_candidates_flat_v,
                        time_step=score_t,
                        alignment=score_alignment,
                        pos_weight=pos_weight,
                        type_weight=type_weight,
                        family_reduce=family_reduce,
                        return_components=False,
                    )
                    valid_candidate_mask = score_candidates > -2
                    masked_scores = torch.where(
                        valid_candidate_mask,
                        score_candidates,
                        torch.full_like(score_candidates, float('inf')),
                    )
                    best_idx = torch.argmin(masked_scores, dim=0)
                    has_valid_candidate = valid_candidate_mask.any(dim=0)
                    debug_selected_scores = torch.where(
                        has_valid_candidate,
                        score_candidates[best_idx, torch.arange(num_graphs, device=ligand_pos.device)],
                        original_log_post,
                    )
                    debug_original_scores = original_log_post
                    debug_valid_candidate_mask = valid_candidate_mask
                    debug_score_label = "selected_log_post"
                graph_idx = torch.arange(num_graphs, device=ligand_pos.device)

                selected_pos = pos_candidates[best_idx, graph_idx].reshape(num_graphs * num_atoms, 3)
                selected_v = v_candidates[best_idx, graph_idx].reshape(num_graphs * num_atoms)
                original_pos = x_prev
                original_v = v_prev
                select_mask = has_valid_candidate.repeat_interleave(num_atoms)
                ligand_pos = torch.where(select_mask.unsqueeze(-1), selected_pos, original_pos)
                ligand_v = torch.where(select_mask, selected_v, original_v)

                if debug_logp:
                    tqdm.write(
                        "[debug_search] "
                        f"t={next_t_scalar} | "
                        f"pos_noise_scale={float(search_pos_noise_scale):.2f} | "
                        f"type_temperature={float(search_type_temperature):.2f} | "
                        f"valid_candidates={int(debug_valid_candidate_mask.sum().item())}/{int(debug_valid_candidate_mask.numel())} | "
                        f"fallback_to_original={int((~has_valid_candidate).sum().item())}/{num_graphs} | "
                        f"original_{debug_score_label}(mean/min/max)=("
                        f"{float(debug_original_scores.mean().item()):.4f}/"
                        f"{float(debug_original_scores.min().item()):.4f}/"
                        f"{float(debug_original_scores.max().item()):.4f}) | "
                        f"{debug_score_label}(mean/min/max)=("
                        f"{float(debug_selected_scores.mean().item()):.4f}/"
                        f"{float(debug_selected_scores.min().item()):.4f}/"
                        f"{float(debug_selected_scores.max().item()):.4f})"
                    )
            else:
                ligand_pos = x_prev
                ligand_v = v_prev

            if use_evaluate:
                score_out = self._estimate_log_p_xref_given_xt_generated_from_alignment(
                    xt_pos=ligand_pos,
                    xt_v=ligand_v,
                    time_step=score_t,
                    alignment=score_alignment,
                    pos_weight=pos_weight,
                    type_weight=type_weight,
                    family_reduce=family_reduce,
                    return_components=debug_logp,
                )
                if debug_logp:
                    log_q_ref = score_out['log_q_ref']
                    log_p_xt = score_out['log_p_xt']
                    log_p_post = score_out['log_post']

                    def _summ(v):
                        return (
                            float(v.mean().item()),
                            float(v.min().item()),
                            float(v.max().item()),
                        )

                    ref_pos_m, ref_pos_min, ref_pos_max = _summ(score_out['ref_pos_log_q'])
                    ref_type_m, ref_type_min, ref_type_max = _summ(score_out['ref_type_log_q'])
                    post_m, post_min, post_max = _summ(log_p_post)
                    tqdm.write(
                        "[debug_logp] "
                        f"t={next_t_scalar} | "
                        f"ref_pos(mean/min/max)=({ref_pos_m:.4f}/{ref_pos_min:.4f}/{ref_pos_max:.4f}) | "
                        f"ref_type(mean/min/max)=({ref_type_m:.4f}/{ref_type_min:.4f}/{ref_type_max:.4f}) | "
                        f"log_post(mean/min/max)=({post_m:.4f}/{post_min:.4f}/{post_max:.4f})"
                    )
                else:
                    log_q_ref, log_p_xt, log_p_post = score_out
            else:
                nan_val = torch.full((num_graphs,), float('nan'), device=ligand_pos.device)
                log_q_ref, log_p_xt, log_p_post = nan_val, nan_val.clone(), nan_val.clone()

            log_q_xt_given_xref_traj.append(log_q_ref.detach().cpu())
            log_p_xt_mc_traj.append(log_p_xt.detach().cpu())
            log_p_xref_given_xt_traj.append(log_p_post.detach().cpu())
            score_time_steps.append(next_t_scalar)

            if store_trajectories:
                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())

            if center_pos_mode != 'none':
                ori_ligand_pos = ligand_pos + offset[batch_ligand]
            else:
                ori_ligand_pos = ligand_pos

            if store_trajectories:
                pos_traj.append(ori_ligand_pos.clone().cpu())
                v_traj.append(ligand_v.clone().cpu())

        if center_pos_mode != 'none':
            ligand_pos = ligand_pos + offset[batch_ligand]

        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'bond': pred_bonds,
            'pos_traj': pos_traj,
            'pos_cond_traj': pos_cond_traj,
            'pos_uncond_traj': pos_uncond_traj,
            'v_traj': v_traj,
            'v_cond_traj': v_cond_traj,
            'v_uncond_traj': v_uncond_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,
            'log_q_xt_given_xref_traj': log_q_xt_given_xref_traj,
            'log_p_xt_mc_traj': log_p_xt_mc_traj,
            'log_p_xref_given_xt_traj': log_p_xref_given_xt_traj,
            'guidance_eval_steps': score_time_steps if eval_step_set is None else sorted(eval_step_set, reverse=True),
            'score_time_steps': score_time_steps,
        }

    @torch.no_grad()
    def sample_scaffold_with_path_logprob(
        self,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        pocket_data,
        scaffold_mask,
        fixed_ligand_pos,
        fixed_ligand_v,
        ligand_shape=None,
        mask_shape_emb=None,
        num_steps=None,
        center_pos_mode=None,
        pred_bond=False,
    ):
        """
        Sample scaffold as usual and additionally return a relative log-prob proxy of the
        reverse sampling path, which can be used as Monte Carlo weights across sampled x0's.
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        num_graphs = batch_ligand.max().item() + 1
        scaffold_mask = torch.as_tensor(scaffold_mask, device=batch_ligand.device).view(-1)
        if scaffold_mask.numel() != batch_ligand.numel():
            raise ValueError(
                f"scaffold_mask size mismatch: got {scaffold_mask.numel()}, expected {batch_ligand.numel()}"
            )

        generated_mask = scaffold_mask == 1
        fixed_mask = ~generated_mask
        generated_float = generated_mask.float()

        if init_ligand_v.dim() != 1:
            raise ValueError("sample_scaffold_with_path_logprob currently expects categorical ligand_v (1D indices).")

        fixed_ligand_pos = torch.as_tensor(
            fixed_ligand_pos, device=init_ligand_pos.device, dtype=init_ligand_pos.dtype
        )
        fixed_ligand_v = torch.as_tensor(
            fixed_ligand_v, device=init_ligand_v.device, dtype=init_ligand_v.dtype
        )
        if fixed_ligand_pos.shape != init_ligand_pos.shape:
            raise ValueError(
                f"fixed_ligand_pos shape mismatch: got {fixed_ligand_pos.shape}, expected {init_ligand_pos.shape}"
            )
        if fixed_ligand_v.shape != init_ligand_v.shape:
            raise ValueError(
                f"fixed_ligand_v shape mismatch: got {fixed_ligand_v.shape}, expected {init_ligand_v.shape}"
            )

        init_ligand_pos, offset = center_pos(init_ligand_pos, batch_ligand, mode=center_pos_mode)
        if center_pos_mode != 'none':
            fixed_pos_centered = fixed_ligand_pos - offset[batch_ligand]
        else:
            fixed_pos_centered = fixed_ligand_pos

        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        ligand_pos[fixed_mask] = fixed_pos_centered[fixed_mask]
        ligand_v[fixed_mask] = fixed_ligand_v[fixed_mask]

        pos_traj, v_traj = [], []
        pos_cond_traj, v_cond_traj = [], []
        pos_uncond_traj, v_uncond_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        pred_bonds = None
        path_logprob = torch.zeros(num_graphs, device=ligand_pos.device, dtype=ligand_pos.dtype)

        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)
            preds_with_cond = self(
                ligand_pos_perturbed=ligand_pos,
                ligand_v_perturbed=ligand_v,
                batch_ligand=batch_ligand,
                pocket_data=pocket_data,
                ligand_shape=ligand_shape,
                mask_shape_emb=mask_shape_emb,
                if_test=True,
                pred_bond=True if self.pred_bond_type and pred_bond and i == 0 else False,
                time_step=t,
            )

            if self.pred_bond_type and pred_bond and i == 0:
                pred_bonds = torch.argmax(preds_with_cond['bond_pred'], dim=-1).view(-1, 1)
                pred_edge_indices = preds_with_cond['edge_index'].transpose(0, 1)
                pred_bonds = torch.cat([pred_edge_indices, pred_bonds], dim=1)

            preds = preds_with_cond
            pos_cond_traj.append(preds_with_cond['pred_ligand_pos'])
            v_cond_traj.append(preds_with_cond['pred_ligand_v'])

            if self.v_mode == 'tomask':
                preds['pred_ligand_v'][:, -1] = -1.e5

            pos0_from_e = preds['pred_ligand_pos']
            v0_from_e = preds['pred_ligand_v']

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            pos_std = (0.5 * pos_log_variance).exp()
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            noise_pos = torch.randn_like(ligand_pos)
            x_prev = pos_model_mean + nonzero_mask * pos_std * noise_pos

            if i > 0:
                pos_logprob_atom = (-0.5 * noise_pos.pow(2).sum(dim=-1)) * generated_float
                path_logprob = path_logprob + scatter_sum(pos_logprob_atom, batch_ligand, dim=0)

            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes + int(self.v_mode == 'tomask'))
            log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
            v_prev = log_sample_categorical(log_model_prob)

            log_v_prev = index_to_log_onehot(v_prev, self.num_classes + int(self.v_mode == 'tomask'))
            v_logprob_atom = log_categorical(log_v_prev, log_model_prob) * generated_float
            path_logprob = path_logprob + scatter_sum(v_logprob_atom, batch_ligand, dim=0)

            x_prev[fixed_mask] = fixed_pos_centered[fixed_mask]
            v_prev[fixed_mask] = fixed_ligand_v[fixed_mask]

            ligand_pos = x_prev
            ligand_v = v_prev

            v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
            vt_pred_traj.append(log_model_prob.clone().cpu())

            if center_pos_mode != 'none':
                ori_ligand_pos = ligand_pos + offset[batch_ligand]
            else:
                ori_ligand_pos = ligand_pos

            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        if center_pos_mode != 'none':
            ligand_pos = ligand_pos + offset[batch_ligand]

        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'bond': pred_bonds,
            'pos_traj': pos_traj,
            'pos_cond_traj': pos_cond_traj,
            'pos_uncond_traj': pos_uncond_traj,
            'v_traj': v_traj,
            'v_cond_traj': v_cond_traj,
            'v_uncond_traj': v_uncond_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,
            'path_logprob': path_logprob,
        }

    @torch.no_grad()
    def sample_diffusion_scaffold_DPS(
        self,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        pocket_data,
        fixed_mask,
        fixed_ligand_pos,
        fixed_ligand_v,
        ligand_shape=None,
        mask_shape_emb=None,
        num_steps=None,
        center_pos_mode=None,
        pred_bond=False,
        refine_steps=1,
        pca_perturb=None,
        dps_gamma=1e-1,
        dps_sigma=5.0,
    ):
        if num_steps is None:
            num_steps = self.num_timesteps
        if refine_steps < 1:
            raise ValueError(f"refine_steps must be >= 1, got {refine_steps}")
        if dps_sigma <= 0:
            raise ValueError(f"dps_sigma must be positive, got {dps_sigma}")

        num_graphs = batch_ligand.max().item() + 1
        fixed_mask = torch.as_tensor(fixed_mask, device=batch_ligand.device).view(-1)
        if fixed_mask.numel() != batch_ligand.numel():
            raise ValueError(
                f"fixed_mask size mismatch: got {fixed_mask.numel()}, expected {batch_ligand.numel()}"
            )
        fixed_mask = fixed_mask == 0
        if init_ligand_v.dim() != 1:
            raise ValueError("sample_diffusion_scaffold_DPS currently expects categorical ligand_v (1D indices).")

        fixed_ligand_pos = torch.as_tensor(
            fixed_ligand_pos, device=init_ligand_pos.device, dtype=init_ligand_pos.dtype
        )
        fixed_ligand_v = torch.as_tensor(
            fixed_ligand_v, device=init_ligand_v.device, dtype=init_ligand_v.dtype
        )
        if fixed_ligand_pos.shape != init_ligand_pos.shape:
            raise ValueError(
                f"fixed_ligand_pos shape mismatch: got {fixed_ligand_pos.shape}, expected {init_ligand_pos.shape}"
            )
        if fixed_ligand_v.shape != init_ligand_v.shape:
            raise ValueError(
                f"fixed_ligand_v shape mismatch: got {fixed_ligand_v.shape}, expected {init_ligand_v.shape}"
            )

        if torch.is_tensor(dps_gamma):
            dps_gamma = dps_gamma.to(device=init_ligand_pos.device, dtype=init_ligand_pos.dtype)

        init_ligand_pos, offset = center_pos(init_ligand_pos, batch_ligand, mode=center_pos_mode)
        if center_pos_mode != 'none':
            fixed_pos_centered = fixed_ligand_pos - offset[batch_ligand]
        else:
            fixed_pos_centered = fixed_ligand_pos

        pos_traj, v_traj = [], []
        pos_cond_traj, v_cond_traj = [], []
        pos_uncond_traj, v_uncond_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []

        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        ligand_v[fixed_mask] = fixed_ligand_v[fixed_mask]

        pred_bonds = None
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))

        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)

            x_t_cur = ligand_pos
            v_cur = ligand_v
            preds_with_cond = None
            log_ligand_v_recon = None
            log_model_prob = None

            for u in range(refine_steps):
                with torch.enable_grad():
                    x_t_grad = x_t_cur.detach().clone().requires_grad_(True)
                    preds_with_cond = self(
                        ligand_pos_perturbed=x_t_grad,
                        ligand_v_perturbed=v_cur,
                        batch_ligand=batch_ligand,
                        pocket_data=pocket_data,
                        ligand_shape=ligand_shape,
                        mask_shape_emb=mask_shape_emb,
                        if_test=True,
                        pred_bond=True if self.pred_bond_type and pred_bond and i == 0 and u == (refine_steps - 1) else False,
                        time_step=t,
                        pca_perturb=pca_perturb,
                    )

                    preds = preds_with_cond
                    if self.v_mode == 'tomask':
                        preds['pred_ligand_v'][:, -1] = -1.e5

                    pos0_from_e = preds['pred_ligand_pos']
                    v0_from_e = preds['pred_ligand_v']

                    pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=x_t_grad, t=t, batch=batch_ligand)
                    pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                    nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
                    x_prev_unknown = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(x_t_grad)

                    fixed_pos_error = pos0_from_e[fixed_mask] - fixed_pos_centered[fixed_mask]
                    dps_loss = 0.5 * fixed_pos_error.pow(2).sum() / (dps_sigma ** 2)
                    dps_grad = torch.autograd.grad(dps_loss, x_t_grad, only_inputs=True)[0]

                if torch.is_tensor(dps_gamma):
                    if dps_gamma.dim() == 0:
                        gamma_t = dps_gamma
                    elif dps_gamma.numel() == self.num_timesteps:
                        gamma_t = dps_gamma[t][batch_ligand].unsqueeze(-1)
                    else:
                        raise ValueError(
                            f"dps_gamma tensor must be scalar or have length {self.num_timesteps}, got {tuple(dps_gamma.shape)}"
                        )
                else:
                    gamma_t = float(dps_gamma)

                dps_grad_masked = torch.zeros_like(dps_grad)
                dps_grad_masked[fixed_mask] = dps_grad[fixed_mask]
                x_prev_unknown = x_prev_unknown.detach() - gamma_t * dps_grad_masked.detach()

                log_ligand_v_recon = F.log_softmax(v0_from_e.detach(), dim=-1)
                log_v_cur = index_to_log_onehot(v_cur, self.num_classes + int(self.v_mode == 'tomask'))
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_v_cur, t, batch_ligand)
                v_prev_unknown = log_sample_categorical(log_model_prob)

                log_v0_known = index_to_log_onehot(
                    fixed_ligand_v, self.num_classes + int(self.v_mode == 'tomask')
                )
                if i > 0:
                    t_prev = torch.full(size=(num_graphs,), fill_value=i - 1, dtype=torch.long, device=ligand_pos.device)
                    log_v_prev_known = self.q_v_pred(log_v0_known, t_prev, batch_ligand)
                    v_prev_known = log_sample_categorical(log_v_prev_known)
                else:
                    v_prev_known = fixed_ligand_v

                x_prev = x_prev_unknown
                v_prev = v_prev_unknown.clone()
                v_prev[fixed_mask] = v_prev_known[fixed_mask]

                if (u < refine_steps - 1) and (i > 1):
                    beta_t = extract(self.betas, t, batch_ligand)
                    x_t_cur = (1.0 - beta_t).sqrt() * x_prev + beta_t.sqrt() * torch.randn_like(x_prev)
                    log_v_prev = index_to_log_onehot(v_prev, self.num_classes + int(self.v_mode == 'tomask'))
                    log_v_t = self.q_v_pred_one_timestep(log_v_prev, t, batch_ligand)
                    v_cur = log_sample_categorical(log_v_t)
                else:
                    x_t_cur = x_prev
                    v_cur = v_prev

            if self.pred_bond_type and pred_bond and i == 0 and preds_with_cond is not None and 'bond_pred' in preds_with_cond:
                pred_bonds = torch.argmax(preds_with_cond['bond_pred'], dim=-1).view(-1, 1)
                pred_edge_indices = preds_with_cond['edge_index'].transpose(0, 1)
                pred_bonds = torch.cat([pred_edge_indices, pred_bonds], dim=1)

            ligand_pos = x_t_cur
            ligand_v = v_cur

            if preds_with_cond is not None:
                pos_cond_traj.append(preds_with_cond['pred_ligand_pos'].detach())
                v_cond_traj.append(preds_with_cond['pred_ligand_v'].detach())
            if log_ligand_v_recon is not None:
                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
            if log_model_prob is not None:
                vt_pred_traj.append(log_model_prob.clone().cpu())

            if center_pos_mode != 'none':
                ori_ligand_pos = ligand_pos + offset[batch_ligand]
            else:
                ori_ligand_pos = ligand_pos

            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        if center_pos_mode != 'none':
            ligand_pos = ligand_pos + offset[batch_ligand]

        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'bond': pred_bonds,
            'pos_traj': pos_traj,
            'pos_cond_traj': pos_cond_traj,
            'pos_uncond_traj': pos_uncond_traj,
            'v_traj': v_traj,
            'v_cond_traj': v_cond_traj,
            'v_uncond_traj': v_uncond_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj
        }
    
    @torch.no_grad()
    def add_noise_and_denoise(
        self,
        ligand_data,
        pocket_data,
        time_step=None,           
        center_pos_mode=None,    
        pred_bond=False,
        return_traj=False,
    ):
        """
        1) Take clean ligand (x0, v0) from ligand_data.
        2) Add forward noise to reach step t.
        3) Run reverse diffusion from t -> 0.
        4) Return x_t, v_t, and the reconstructed x0_hat, v0_hat (plus optional traj).
        """

        batch_ligand = ligand_data.batch
        ligand_pos = ligand_data.ligand_coordinate         
        ligand_v = ligand_data.ligand_atom_feature_full    
        ligand_bond_index = getattr(ligand_data, "ligand_bond_index", None)
        ligand_bond_type  = getattr(ligand_data, "ligand_bond_type",  None)

        num_graphs = batch_ligand.max().item() + 1

        if center_pos_mode is None:
            center_pos_mode = getattr(self, "center_pos_mode", "none")

        ligand_pos, offset = center_pos(ligand_pos, batch_ligand, mode=center_pos_mode)

        if time_step is None:
            if hasattr(self, "sample_time"):
                time_step, _ = self.sample_time(num_graphs, ligand_pos.device)
            else:
                time_step = torch.randint(
                    low=0,
                    high=self.num_timesteps,
                    size=(num_graphs,),
                    device=ligand_pos.device,
                )
        else:
            if not torch.is_tensor(time_step):
                time_step = torch.full(
                    (num_graphs,),
                    int(time_step),
                    dtype=torch.long,
                    device=ligand_pos.device,
                )

        assert torch.all(time_step == time_step[0]), \
            "This helper currently expects a single common t for the whole batch."
        t0 = int(time_step[0].item())

        # ---------- forward: q(x_t | x_0), q(v_t | v_0) ----------
        # positions
        a = self.alphas_cumprod.index_select(0, time_step)          
        a_pos = a[batch_ligand].unsqueeze(-1)                       
        pos_noise = torch.randn_like(ligand_pos)
        ligand_pos_t = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise

        # atom types
        if self.v_mode == "uniform":
            log_v0 = index_to_log_onehot(ligand_v, self.num_classes)
            ligand_v_t, log_vt = self.q_v_sample(
                log_v0, time_step, batch_ligand, self.num_classes
            )
            num_classes_eff = self.num_classes
        elif self.v_mode == "tomask":
            log_v0 = index_to_log_onehot(ligand_v, self.num_classes + 1)
            ligand_v_t, log_vt = self.q_v_sample(
                log_v0, time_step, batch_ligand, self.num_classes + 1
            )
            num_classes_eff = self.num_classes + 1
        else:
            raise ValueError(f"Unsupported v_mode: {self.v_mode}")

        ligand_shape = None
        mask_shape_emb = None

        ligand_pos_cur = ligand_pos_t.clone()
        ligand_v_cur = ligand_v_t.clone()
        pred_bonds = None

        pos_traj, v_traj = [], []

        for tt in tqdm(reversed(range(t0 + 1))):
            t = torch.full(
                (num_graphs,),
                tt,
                dtype=torch.long,
                device=ligand_pos.device,
            )

            # model predicts x0 (and v0) from current (x_t, v_t)
            preds = self(
                ligand_pos_perturbed=ligand_pos_cur,
                ligand_v_perturbed=ligand_v_cur,
                batch_ligand=batch_ligand,
                pocket_data=pocket_data,
                ligand_shape=ligand_shape,
                ligand_bond_index=ligand_bond_index,
                ligand_bond_type=ligand_bond_type,
                mask_shape_emb=mask_shape_emb,
                time_step=t,
                if_test=True,
                pred_bond=bool(self.pred_bond_type and pred_bond and tt == 0),
            )

            if self.pred_bond_type and pred_bond and tt == 0:
                bond_type_pred = torch.argmax(preds["bond_pred"], dim=-1).view(-1, 1)
                edge_index = preds["edge_index"].t()  
                pred_bonds = torch.cat([edge_index, bond_type_pred], dim=1)

            if self.v_mode == "tomask":
                preds["pred_ligand_v"][:, -1] = -1e5

            # ---- position update: q(x_{t-1} | x_t, x0_hat) ----
            pos0_hat = preds["pred_ligand_pos"]
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_hat,
                xt=ligand_pos_cur,
                t=t,
                batch=batch_ligand,
            )
            pos_logvar = extract(self.posterior_logvar, t, batch_ligand)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)

            noise = torch.randn_like(ligand_pos_cur)
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_logvar).exp() * noise

            # ---- type update: q(v_{t-1} | v_t, v0_hat) ----
            log_v0_hat = F.log_softmax(preds["pred_ligand_v"], dim=-1)
            log_v_cur = index_to_log_onehot(ligand_v_cur, num_classes_eff)
            log_model_prob = self.q_v_posterior(
                log_v0_hat,
                log_v_cur,
                t,
                batch_ligand,
            )
            ligand_v_next = log_sample_categorical(log_model_prob)

            ligand_pos_cur = ligand_pos_next
            ligand_v_cur = ligand_v_next

            if center_pos_mode != "none":
                pos_out = ligand_pos_cur + offset[batch_ligand]
            else:
                pos_out = ligand_pos_cur

            if return_traj:
                pos_traj.append(pos_out.cpu())
                v_traj.append(ligand_v_cur.cpu())

        if center_pos_mode != "none":
            ligand_pos_cur = ligand_pos_cur + offset[batch_ligand]

        out = {
            "t0": t0,
            "xt_pos": ligand_pos_t if center_pos_mode == "none" else ligand_pos_t + offset[batch_ligand],
            "xt_v": ligand_v_t,
            "x0_pos_pred": ligand_pos_cur,
            "x0_v_pred": ligand_v_cur,
        }
        if pred_bonds and self.pred_bond_type:
            out["bond"] = pred_bonds
        if return_traj:
            out["pos_traj"] = pos_traj
            out["v_traj"] = v_traj

        return out





class ScorePosNet3D_opt(ScorePosNet3D):
    def __init__(self, config, ligand_atom_feature_dim, ligand_bond_feature_dim):
        super(ScorePosNet3D_opt, self).__init__(config, ligand_atom_feature_dim, ligand_bond_feature_dim)
    
    @torch.no_grad()
    def sample_diffusion(self, init_ligand_pos, init_ligand_v, batch_ligand, pocket_data, ligand_shape=None, 
                         mask_shape_emb=None, num_steps=None, center_pos_mode=None, use_mesh_data=None, use_pointcloud_data=None, 
                         use_pocket_data=None, grad_step=500, pred_bond=False, gaussian_noise_traj=None, gumbel_noise_traj=None):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_ligand.max().item() + 1
        print('sample center pos mode: ', center_pos_mode)


        init_ligand_pos, offset = center_pos(init_ligand_pos, batch_ligand, mode=center_pos_mode)
        
        
        pos_traj, v_traj = [], []
        pos_cond_traj, v_cond_traj = [], []
        pos_uncond_traj, v_uncond_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        pred_bonds = None
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        idx = 0

        for i in tqdm(time_seq, desc='sampling', total=len(time_seq), disable=True):
        # for i in time_seq:
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)
            with torch.no_grad():
                
                preds_with_cond = self(
                    ligand_pos_perturbed=ligand_pos,
                    ligand_v_perturbed=ligand_v,
                    batch_ligand=batch_ligand,
                    pocket_data=pocket_data,
                    ligand_shape=ligand_shape,
                    mask_shape_emb=mask_shape_emb,
                    if_test=True,
                    pred_bond=True if self.pred_bond_type and pred_bond and i == 0 else False,
                    time_step=t
                )

            if self.pred_bond_type and pred_bond and i == 0:
                pred_bonds = torch.argmax(preds_with_cond['bond_pred'], dim=-1).view(-1, 1)
                pred_edge_indices = preds_with_cond['edge_index'].transpose(0, 1)
                pred_bonds = torch.cat([pred_edge_indices, pred_bonds], dim=1)

            preds = preds_with_cond
                

            pos_cond_traj.append(preds_with_cond['pred_ligand_pos'])
            v_cond_traj.append(preds_with_cond['pred_ligand_v'])
            
            if self.v_mode == 'tomask':
                preds['pred_ligand_v'][:, -1] = -1.e5


            with torch.no_grad():
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
                
                pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                # no noise when t == 0
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * gaussian_noise_traj[idx]

                ligand_pos = ligand_pos_next

                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes+int(self.v_mode=='tomask'))
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)

                ligand_v_next = log_sample_categorical(log_model_prob, gumbel_noise = gumbel_noise_traj[idx])
                idx += 1


                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

                if center_pos_mode != 'none':
                    ori_ligand_pos = ligand_pos + offset[batch_ligand]
                else:
                    ori_ligand_pos = ligand_pos
                
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())
    
        if center_pos_mode != 'none':
            ligand_pos = ligand_pos + offset[batch_ligand]
        
        if pred_bond:

            return {
                'pos': ligand_pos,
                'v': ligand_v,
                'bond': pred_bonds,
                'pos_traj': pos_traj,
                'pos_cond_traj': pos_cond_traj,
                'pos_uncond_traj': pos_uncond_traj,
                'v_traj': v_traj,
                'v_cond_traj': v_cond_traj,
                'v_uncond_traj': v_uncond_traj,
                'v0_traj': v0_pred_traj,
                'vt_traj': vt_pred_traj
            }
        else:
            return {
                'pos': ligand_pos,
                'v': ligand_v,
                'bond': pred_bonds,
                'pos_traj': pos_traj,
                'pos_cond_traj': pos_cond_traj,
                'pos_uncond_traj': pos_uncond_traj,
                'v_traj': v_traj,
                'v_cond_traj': v_cond_traj,
                'v_uncond_traj': v_uncond_traj,
                'v0_traj': v0_pred_traj,
                'vt_traj': vt_pred_traj
            }
