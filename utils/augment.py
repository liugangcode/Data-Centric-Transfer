import time
import torch
from tqdm import trange
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset

from .sde import VESDE
from .infonce import InfoNCE
from .solver import LangevinCorrector, ReverseDiffusionPredictor, mask_x, mask_adjs, gen_noise, get_score_fn

from .mol_utils import convert_dense_to_rawpyg, convert_sparse_to_dense, combine_graph_inputs, convert_dense_adj_to_sparse_with_attr
from .mol_utils import extract_graph_feature, estimate_feature_embs

__all__ = ['build_augmentation_dataset']

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
reg_criterion = torch.nn.MSELoss(reduction='none')

# -------- get negative samples for infoNCE loss --------
def get_negative_indices(y_true, n_sample=10):
    if torch.isnan(y_true).sum() != 0:
        print('y_true', (y_true==y_true).size(), (y_true==y_true).sum())
        return None
    y_true = torch.nan_to_num(y_true, nan=0.)
    task_num = y_true.size(1)
    diffs = torch.abs(y_true.view(-1,1,task_num) - y_true.view(1,-1,task_num)).mean(dim=-1)
    diffs_desc_indices = torch.argsort(diffs, dim=1, descending=True)
    return diffs_desc_indices[:, :n_sample]


def inner_sampling(args, generator, x, adj, sde_x, sde_adj, diff_steps, flags=None):
    score_fn_x = get_score_fn(sde_x, generator['model_x'], perturb_ratio=args.perturb_ratio)
    score_fn_adj = get_score_fn(sde_adj, generator['model_adj'], perturb_ratio=args.perturb_ratio)
    snr, scale_eps, n_steps= args.snr, args.scale_eps, args.n_steps
    predictor_obj_x = ReverseDiffusionPredictor('x', sde_x, score_fn_x, False, perturb_ratio=args.perturb_ratio)
    corrector_obj_x = LangevinCorrector('x', sde_x, score_fn_x, snr, scale_eps, n_steps, perturb_ratio=args.perturb_ratio)
    predictor_obj_adj = ReverseDiffusionPredictor('adj', sde_adj, score_fn_adj, False, perturb_ratio=args.perturb_ratio)
    corrector_obj_adj = LangevinCorrector('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps, perturb_ratio=args.perturb_ratio)
    x, adj = mask_x(x, flags), mask_adjs(adj, flags)
    total_sample_steps = args.out_steps
    timesteps = torch.linspace(1, 1e-3, total_sample_steps, device=args.device)[-diff_steps:]
    with torch.no_grad():
        # -------- Reverse diffusion process --------
        for i in range(diff_steps):
            t = timesteps[i]
            vec_t = torch.ones(adj.shape[0], device=t.device) * t
            _x = x
            x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
            adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)
            _x = x
            x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
            adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
    return x_mean.detach(), adj_mean.detach()

## -------- Main function for augmentation--------##
def build_augmentation_dataset(args, model, generator, labeled_data):
    if args.dataset.startswith('nx'):
        raise NotImplementedError(f"currently not implemented.")

    infonce_paired = InfoNCE(temperature=0.05)
    if 'classification' in args.task_type:
        criterion = cls_criterion
        prob_func = torch.nn.Sigmoid()
    else:
        criterion = reg_criterion

    label_split_idx = labeled_data.get_idx_split()
    kept_pyg_list = []
    augmented_pyg_list = []
    augment_fails = 0

    labeled_trainloader = DataLoader(labeled_data[label_split_idx["train"]], batch_size=args.aug_batch, shuffle=False, num_workers = 0)

    for step, batch_data in enumerate(labeled_trainloader):
        model.eval()
        batch_data_list = batch_data.to_data_list()
        batch_data = batch_data.to(args.device)
        if batch_data.x.shape[0] > 1:
            with torch.no_grad():
                y_pred_logits = model(batch_data)[0]
            y_true_all, batch_index = batch_data.y.to(torch.float32), batch_data.batch
            is_labeled = y_true_all == y_true_all
            y_pred_logits[~is_labeled], y_true_all[~is_labeled] = 0, 0

            selected_topk = args.topk
            if args.topk > y_true_all.size(0):
                selected_topk = y_true_all.size(0)
                print('reset topk to: ', selected_topk)

            topk_indices = torch.topk(criterion(y_pred_logits.view(y_true_all.size()).to(torch.float32), y_true_all).view(y_true_all.size()).sum(dim=-1), selected_topk, largest=False, sorted=True).indices
            augment_mask = torch.zeros(y_pred_logits.size(0)).to(y_pred_logits.device).scatter_(0, topk_indices, 1).bool()
            augment_labels = y_true_all[augment_mask]

            if args.strategy.split('_')[0] == 'add':
                kept_indices = list(range(y_pred_logits.size(0)))
            elif args.strategy.split('_')[0] == 'replace':
                kept_indices = (~augment_mask).nonzero().view(-1).cpu().tolist()
            else:
                raise NotImplementedError(f"not implemented strategy {args.strategy}.")
            for kept_index in kept_indices:
                kept_pyg_list.append(batch_data_list[kept_index])

            batch_dense_x, batch_dense_x_disable, batch_dense_enable_adj, batch_dense_disable_adj, batch_node_mask = \
                convert_sparse_to_dense(batch_index, batch_data.x, batch_data.edge_index, batch_data.edge_attr, augment_mask=None)

            if batch_dense_x_disable is None:
                ori_x, ori_adj = batch_dense_x.clone(), batch_dense_enable_adj.clone()
            else:
                ori_x, ori_adj = combine_graph_inputs(*convert_sparse_to_dense(batch_index, batch_data.x, batch_data.edge_index, batch_data.edge_attr, augment_mask=None, return_node_mask=False))
            ori_x, ori_adj = ori_x.to(torch.float32), ori_adj.to(torch.float32)

            ## extract graph feature
            ori_feats  = extract_graph_feature(ori_x, ori_adj, node_mask=batch_node_mask)

            neg_indices = get_negative_indices(y_true_all, n_sample=args.n_negative) # B x n_sample
            neg_indices = neg_indices[augment_mask]
        
            # sde
            total_sample_steps = args.out_steps
            sde_x = VESDE(sigma_min=0.1, sigma_max=1, N=total_sample_steps)
            sde_adj = VESDE(sigma_min=0.1, sigma_max=1, N=total_sample_steps)
            
    
            batch_dense_x, batch_dense_enable_adj = batch_dense_x[augment_mask], batch_dense_enable_adj[augment_mask]
            if batch_dense_x_disable is not None:
                batch_dense_x_disable, batch_dense_disable_adj = batch_dense_x_disable[augment_mask], batch_dense_disable_adj[augment_mask]
            
            # perturb x 
            peturb_t = torch.ones(batch_dense_enable_adj.shape[0]).to(args.device) * (sde_adj.T - 1e-3) + 1e-3
            mean_x, std_x = sde_x.marginal_prob(batch_dense_x, peturb_t)
            z_x = gen_noise(batch_dense_x, flags=batch_node_mask[augment_mask], sym=False, perturb_ratio=args.perturb_ratio)
            perturbed_x = mean_x + std_x[:, None, None] * z_x
            perturbed_x = mask_x(perturbed_x, batch_node_mask[augment_mask])
            
            # perturb adj
            mean_adj, std_adj = sde_adj.marginal_prob(batch_dense_enable_adj, peturb_t)
            z_adj = gen_noise(batch_dense_enable_adj, flags=batch_node_mask[augment_mask], sym=True, perturb_ratio=args.perturb_ratio)
            perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
            perturbed_adj = mask_adjs(perturbed_adj, batch_node_mask[augment_mask])
            
            timesteps = torch.linspace(1, 1e-3, total_sample_steps, device=args.device)[-args.out_steps:]
            def get_aug_grads(prediction_model, inner_output_data):
                prediction_model.eval()
                inner_output_x, inner_output_adj = inner_output_data
                inner_output_adj = mask_adjs(inner_output_adj, batch_node_mask[augment_mask])
                inner_output_x = mask_x(inner_output_x, batch_node_mask[augment_mask])

                inner_output_x,  inner_output_adj = inner_output_x.requires_grad_(), inner_output_adj.requires_grad_()
                with torch.enable_grad():
                    if batch_dense_x_disable is None:
                        inner_x_all, inner_adj_all = inner_output_x, inner_output_adj
                    else:
                        inner_x_all, inner_adj_all = combine_graph_inputs(inner_output_x, batch_dense_x_disable, inner_output_adj, batch_dense_disable_adj, mode='continuous')

                    edge_index, edge_attr = convert_dense_adj_to_sparse_with_attr(inner_adj_all, batch_node_mask[augment_mask])
                    bdata_batch_index = batch_node_mask[augment_mask].nonzero()[:,0]
                    bdata_y = augment_labels.view(inner_x_all.size(0), -1)
                    node_feature_encoded = estimate_feature_embs(inner_x_all[batch_node_mask[augment_mask]], batch_data, prediction_model, obj='node')
                    edge_attr_encoded = estimate_feature_embs(edge_attr, batch_data, prediction_model, obj='edge')
                    bdata = Data(x=node_feature_encoded, edge_index=edge_index, edge_attr=edge_attr_encoded, y=bdata_y, batch=bdata_batch_index)
                    
                    if inner_output_x.shape[0] > 1:
                        preds = prediction_model(bdata, encode_raw=False)[0]
                        is_labeled = bdata.y == bdata.y
                        bdata_target = bdata.y.to(torch.float32)[is_labeled]
                        if 'classification' in args.task_type:
                            bdata_probs = prob_func(preds.view(bdata.y.size()).to(torch.float32)[is_labeled])
                            loss_y = torch.log((bdata_probs * bdata_target + (1 - bdata_probs) * (1 - bdata_target)).clamp_min(1e-18)).mean() # maximize log likelihood
                        else:
                            loss_y = - reg_criterion(preds.view(bdata.y.size()).to(torch.float32)[is_labeled], bdata_target).mean()
                        
                        aug_feats = extract_graph_feature(inner_x_all, inner_adj_all, node_mask=batch_node_mask[augment_mask])
                        
                        query_structure_embed, pos_structure_embed, neg_structure_embed = aug_feats, ori_feats[augment_mask], ori_feats[neg_indices]
                        loss_structure = infonce_paired(query_structure_embed, pos_structure_embed, neg_structure_embed) # maximize infonce == minimize mutual information                        

                        total_loss =  loss_y + loss_structure
    
                        aug_grad_x, aug_grad_adj = torch.autograd.grad(total_loss, [inner_output_x, inner_output_adj])
                    else:
                        aug_grad_x, aug_grad_adj = None, None
                return aug_grad_x, aug_grad_adj

            score_fn_x = get_score_fn(sde_x, generator['model_x'], perturb_ratio=args.perturb_ratio)
            score_fn_adj = get_score_fn(sde_adj, generator['model_adj'], perturb_ratio=args.perturb_ratio)
            predictor_obj_x = ReverseDiffusionPredictor('x', sde_x, score_fn_x, False, perturb_ratio=args.perturb_ratio)
            corrector_obj_x = LangevinCorrector('x', sde_x, score_fn_x, args.snr, args.scale_eps, args.n_steps, perturb_ratio=args.perturb_ratio)
            predictor_obj_adj = ReverseDiffusionPredictor('adj', sde_adj, score_fn_adj, False, perturb_ratio=args.perturb_ratio)
            corrector_obj_adj = LangevinCorrector('adj', sde_adj, score_fn_adj, args.snr, args.scale_eps, args.n_steps, perturb_ratio=args.perturb_ratio)
            if args.no_print:
                outer_iters = range(args.out_steps)
            else:
                outer_iters = trange(0, (args.out_steps), desc = '[Outer Sampling]', position = 1, leave=False)
            for i in outer_iters:
                inner_output_x, inner_output_adj = inner_sampling(args, generator, perturbed_x, perturbed_adj, sde_x, sde_adj, args.out_steps-i, batch_node_mask[augment_mask])
                aug_grad_x, aug_grad_adj = get_aug_grads(model, [inner_output_x, inner_output_adj])
                with torch.no_grad():
                    t = timesteps[i]
                    vec_t = torch.ones(perturbed_adj.shape[0], device=t.device) * t
                    _x = perturbed_x
                    perturbed_x, perturbed_x_mean = corrector_obj_x.update_fn(perturbed_x, perturbed_adj, batch_node_mask[augment_mask], vec_t, aug_grad=aug_grad_x)
                    perturbed_adj, perturbed_adj_mean = corrector_obj_adj.update_fn(_x, perturbed_adj, batch_node_mask[augment_mask], vec_t, aug_grad=aug_grad_adj)
                    _x = perturbed_x
                    perturbed_x, perturbed_x_mean = predictor_obj_x.update_fn(perturbed_x, perturbed_adj, batch_node_mask[augment_mask], vec_t, aug_grad=aug_grad_x)
                    perturbed_adj, perturbed_adj_mean = predictor_obj_adj.update_fn(_x, perturbed_adj, batch_node_mask[augment_mask], vec_t, aug_grad=aug_grad_adj)

            perturbed_adj_mean = mask_adjs(perturbed_adj_mean, batch_node_mask[augment_mask])
            perturbed_x_mean = mask_x(perturbed_x_mean, batch_node_mask[augment_mask])

            augmented_x, augmented_adj = perturbed_x_mean.cpu(), perturbed_adj_mean.cpu()
            if batch_dense_x_disable is None:
                augmented_x, augmented_adj = augmented_x.clone(), augmented_adj.clone()
            else:
                augmented_x, augmented_adj = combine_graph_inputs(augmented_x, batch_dense_x_disable, augmented_adj, batch_dense_disable_adj, mode='discrete')
            
            batch_augment_pyg_list = convert_dense_to_rawpyg(augmented_x, augmented_adj, augment_labels, n_jobs=args.n_jobs)

            augment_indices = augment_mask.nonzero().view(-1).cpu().tolist()
            augmented_pyg_list_temp = []
            for pyg_data in batch_augment_pyg_list:
                if not isinstance(pyg_data, int):
                    augmented_pyg_list_temp.append(pyg_data)
                elif args.strategy.split('_')[0] == 'add':
                    pass
                else:
                    augment_fails += 1
                    kept_pyg_list.append(batch_data_list[augment_indices[pyg_data]])

            augmented_pyg_list.extend(augmented_pyg_list_temp)
    
    
    kept_pyg_list.extend(augmented_pyg_list)
    new_dataset = NewDataset(kept_pyg_list, num_fail=augment_fails)
    return new_dataset


class NewDataset(InMemoryDataset):
    def __init__(self, data_list, num_fail=0, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data_list = data_list
        self.data_len = len(data_list)
        self.num_fail = num_fail
        # print('data_len', self.data_len, 'num_fail', num_fail)
        self.data, self.slices = self.collate(data_list)
    def get_idx_split(self):
        return {'train': torch.arange(self.data_len, dtype = torch.long), 'valid': None, 'test': None}

if __name__ == '__main__':
    pass