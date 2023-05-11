import torch
import numpy as np
import abc

from .sde import VESDE
from .models.ScoreNetwork_A import ScoreNetworkA
from .models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH

def load_generator(device, path=None):
    device = f'cuda:{device[0]}' if isinstance(device, list) else device
    print('loading to device: ', device)
    if path is None:
        raise Exception("Please specify the path to load the diffusion model.")
    ckpt = torch.load(path, map_location=device)
    denoise_x = load_model_from_ckpt(ckpt['params_x'], ckpt['x_state_dict'], device)
    denoise_adj = load_model_from_ckpt(ckpt['params_adj'], ckpt['adj_state_dict'], device)
    print(f'denoising model from {path} loaded')
    generator = {"model_x": denoise_x, "model_adj": denoise_adj}
    return generator

# -------- score function --------
def get_score_fn(sde, model, perturb_ratio=None):
    model.eval()
    model_fn = model
    if isinstance(sde, VESDE):
        def score_fn(x, adj, flags, t):
            score = model_fn(x, adj, flags)
            if perturb_ratio is None:
                return score / score.size(1)
            else:
                return score * perturb_ratio
    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")
    return score_fn


# -------- Generate noise --------
def gen_noise(x, flags, sym=True, perturb_ratio=None):
    z = torch.randn_like(x)
    if sym:
        z = z.triu(1)
        z = z + z.transpose(-1,-2)
        z = mask_adjs(z, flags)
    else:
        z = mask_x(z, flags)
    if perturb_ratio is None:
        return z / z.size(1) 
    else:
        return z * perturb_ratio

# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):

    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:,:,None]


# -------- Mask batch of adjacency matrices with 0-1 flags tensor --------
def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    :return:
    """
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""
    def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        pass


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, obj, sde, score_fn, probability_flow=False, perturb_ratio = None):
        super().__init__(sde, score_fn, probability_flow)
        self.obj = obj
        self.p_ratio = perturb_ratio


    def update_fn(self, x, adj, flags, t, aug_grad=None):

        if self.obj == 'x':
            f, G = self.rsde.discretize(x, adj, flags, t, is_adj=False, aug_grad=aug_grad)
            z = gen_noise(x, flags, sym=False, perturb_ratio=self.p_ratio)
            x_mean = x - f
            x = x_mean + G[:, None, None] * z
            return x, x_mean

        elif self.obj == 'adj':
            f, G = self.rsde.discretize(x, adj, flags, t, is_adj=True, aug_grad=aug_grad)
            z = gen_noise(adj, flags, sym=True, perturb_ratio=self.p_ratio)
            adj_mean = adj - f
            adj = adj_mean + G[:, None, None] * z
            return adj, adj_mean

        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported.")


class LangevinCorrector(Corrector):
    def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps, perturb_ratio=None):
        super().__init__(sde, score_fn, snr, scale_eps, n_steps)
        self.obj = obj
        self.p_ratio = perturb_ratio
  
    def combine_grad(self, denoised_grad, aug_grad, t=1):
        ratio = denoised_grad.norm(p=2, dim=[1,2], keepdim=True) / aug_grad.norm(p=2, dim=[1,2], keepdim=True).clamp_min(1e-18)
        aug_grad = ratio * aug_grad
        total_scores = denoised_grad * 0.5 + aug_grad * 0.5
        return total_scores

    def update_fn(self, x, adj, flags, t, aug_grad=None):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        seps = self.scale_eps
        alpha = torch.ones_like(t)

        if self.obj == 'x':
            for i in range(n_steps):
                grad = score_fn(x, adj, flags, t)
                if aug_grad is not None:
                    grad = self.combine_grad(grad, aug_grad, t)
                noise = gen_noise(x, flags, sym=False, perturb_ratio=self.p_ratio)
                grad_norm = torch.norm(grad, p=2, dim=[-2,-1], keepdim=True).mean()
                noise_norm = torch.norm(noise, p=2, dim=[-2,-1], keepdim=True).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None, None] * grad
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
            return x, x_mean

        elif self.obj == 'adj':
            for i in range(n_steps):
                grad = score_fn(x, adj, flags, t)
                if aug_grad is not None:
                    grad = self.combine_grad(grad, aug_grad)
                noise = gen_noise(adj, flags, sym=True, perturb_ratio=self.p_ratio)
                grad_norm = torch.norm(grad, p=2, dim=[-2,-1],keepdim=True).mean()
                noise_norm = torch.norm(noise, p=2, dim=[-2,-1],keepdim=True).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:, None, None] * grad
                adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
            return adj, adj_mean

        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported")


################## utils to load generator ####################
    
def load_model(params):
    params_ = params.copy()
    model_type = params_.pop('model_type', None)
    if model_type == 'ScoreNetworkX':
        model = ScoreNetworkX(**params_)
    elif model_type == 'ScoreNetworkX_GMH':
        model = ScoreNetworkX_GMH(**params_)
    elif model_type == 'ScoreNetworkA':
        model = ScoreNetworkA(**params_)
    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")
    return model

def load_model_from_ckpt(params, state_dict, device):
    model = load_model(params)
    if 'module.' in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
    else:
        model = model.to(device)
    return model