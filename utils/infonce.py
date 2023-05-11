import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    '''
    Adapted from: https://github.com/RElbers/info-nce-pytorch 
    '''

    def __init__(self, temperature=0.1, positive_mode='single', distance='cosine'):
        super().__init__()
        self.temperature = temperature
        self.positive_mode = positive_mode
        self.distance = distance

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        positive_mode=self.positive_mode,
                        distance=self.distance)


def info_nce(query, positive_key, negative_keys, temperature=0.1, positive_mode = 'single', distance='cosine'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')

    if positive_mode == 'single' and positive_key.dim() != 2:
        raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'single'.")
    if positive_mode == 'multiple' and negative_keys.dim() != 3:
        raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'multiple'.")
    if negative_keys.dim() != 3:
        raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if query.size(0) != positive_key.size(0):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if len(query) != len(negative_keys):
        raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if query.shape[-1] != negative_keys.shape[-1]:
        raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    
    # Cosine/distance between positive pairs
    positive_logit, negative_logits = calculate_logits(query, positive_key, negative_keys, mode=distance)

    probs = F.softmax(torch.cat([positive_logit, negative_logits], dim=1) / temperature, dim=1)
    positive_probs = probs[:, 0]
    negative_probs = probs[:, 1:]
    return  - torch.log(positive_probs / (negative_probs.sum(dim=1).clamp_min(1e-18)).clamp_min(1e-18) ).mean()

def calculate_logits(query, positive, negative, mode='cosine'):
    positive_logit = calculate_distance(query, positive, mode=mode)
    negative_logits = calculate_distance(query, negative, mode=mode)
    return positive_logit, negative_logits
        
def calculate_distance(query, key, mode='cosine'):
    if key.dim() == 3:
        query = query.unsqueeze(1)
    keepdim = key.dim() == 2
    if mode == 'cosine':
        if key.dim() == 2:
            return torch.sum(query * key, dim=1, keepdim=keepdim)
        else:
            return (query @ transpose(key)).squeeze(1)
    elif mode == 'l1':
        return - torch.norm(query - key, p=1, dim=1, keepdim=keepdim)
    elif mode == 'l2':
        return - torch.norm(query - key, p=2, dim=1, keepdim=keepdim)
    else:
        raise ValueError('Invalid distance mode: {}'.format(mode))

def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]