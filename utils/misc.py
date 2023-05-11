import torch
import numpy as np
from typing import List, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from ogb.graphproppred import Evaluator
from torch_geometric.data import Data, Dataset, InMemoryDataset

__all__ = ['print_info', 'validate', 'init_weights', 'AverageMeter', 'ImbalancedSampler']

class ImbalancedSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(
        self,
        dataset: Union[Dataset, Data, List[Data], torch.Tensor],
        train_index: Optional[torch.Tensor],
    ):
        assert isinstance(dataset, InMemoryDataset)
        y = dataset.data.y.view(-1)[train_index]
        y = y.to(torch.long)
        num_samples = y.numel()
        class_weight = 1. / y.bincount()
        weight = class_weight[y]
        return super().__init__(weight, num_samples, replacement=True)


def print_info(set_name, perf):
    output_str = '{}\t\t'.format(set_name)
    for metric_name in perf.keys():
        output_str += '{}: {:<10.4f} \t'.format(metric_name, perf[metric_name])
    print(output_str)

def validate(args, model, loader):
    y_true = []
    y_pred = []
    device = args.device
    model.eval()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                if args.model == 'grea':
                    pred = model(batch)['pred_rem']
                else:
                    pred = model(batch)[0]
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    if args.task_type in 'regression':
        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mse = mean_squared_error(y_true, y_pred, squared=True)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        perf ={'rmse': rmse, 'mse': mse, 'mae': mae, 'r2': r2}
    else:
        if args.dataset.startswith('ogbg'):
            evaluator = Evaluator(name=args.dataset)
            auc = evaluator.eval({"y_true": y_true, "y_pred": y_pred})['rocauc']
        else:
            assert y_true.shape[1] == y_pred.shape[1]
            auc = _eval_rocauc(y_true, y_pred)['rocauc']
        perf ={'auc': auc}
    return perf


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def log_base(base, x):
    return np.log(x) / np.log(base)

def _eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC averaged across tasks
    '''
    rocauc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')
    return {'rocauc': sum(rocauc_list)/len(rocauc_list)}


if __name__ == '__main__':
    pass