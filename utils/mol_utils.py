import re
import numpy as np
from tqdm import trange
from multiprocessing import Pool
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_scatter import scatter_add, scatter_mean, scatter_min, scatter_max, scatter_std
torch.multiprocessing.set_sharing_strategy('file_system')

from ogb.utils.features import (allowable_features, safe_index) 

# -------- preparation --------
bond_decoder_simple = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC, 5:Chem.rdchem.BondType.UNSPECIFIED}
bond_decoder = {0: Chem.rdchem.BondType.UNSPECIFIED, 1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.QUADRUPLE, 5: Chem.rdchem.BondType.QUINTUPLE, 6: Chem.rdchem.BondType.HEXTUPLE, 7: Chem.rdchem.BondType.ONEANDAHALF, 8: Chem.rdchem.BondType.TWOANDAHALF, 9: Chem.rdchem.BondType.THREEANDAHALF, 10: Chem.rdchem.BondType.FOURANDAHALF, 11: Chem.rdchem.BondType.FIVEANDAHALF, 12: Chem.rdchem.BondType.AROMATIC, 13: Chem.rdchem.BondType.IONIC, 14: Chem.rdchem.BondType.HYDROGEN, 15: Chem.rdchem.BondType.THREECENTER, 16: Chem.rdchem.BondType.DATIVEONE, 17: Chem.rdchem.BondType.DATIVE, 18: Chem.rdchem.BondType.DATIVEL, 19: Chem.rdchem.BondType.DATIVER, 20: Chem.rdchem.BondType.OTHER, 21: Chem.rdchem.BondType.ZERO}

chirality_atom2 = torch.arange(4).to(torch.float32)
degree_atom3 = torch.arange(12).to(torch.float32)
charge_atom4 = torch.arange(12).to(torch.float32)
numH_atom5 = torch.arange(10).to(torch.float32)
number_radical_e_atom6 = torch.arange(6).to(torch.float32)
hybridization_atom7 = torch.arange(6).to(torch.float32)
aromatic_atom8 = torch.arange(2).to(torch.float32)
isinring_atom9 = torch.arange(2).to(torch.float32)
atom_feature_list = [chirality_atom2, degree_atom3, charge_atom4, numH_atom5, number_radical_e_atom6, hybridization_atom7, aromatic_atom8, isinring_atom9]
bondstereo_bond2 = torch.arange(6).to(torch.float32)
isconjugated_bond3 = torch.arange(2).to(torch.float32)
bond_feature_list = [bondstereo_bond2, isconjugated_bond3]
atom_weight_list = torch.tensor([1.0080,   4.0030,   6.9410,   9.0120,  10.8120,  12.0110,  14.0070,
         15.9990,  18.9980,  20.1800,  22.9900,  24.3050,  26.9820,  28.0860,
         30.9740,  32.0670,  35.4530,  39.9480,  39.0980,  40.0780,  44.9560,
         47.8670,  50.9420,  51.9960,  54.9380,  55.8450,  58.9330,  58.6930,
         63.5460,  65.3900,  69.7230,  72.6100,  74.9220,  78.9600,  79.9040,
         83.8000,  85.4680,  87.6200,  88.9060,  91.2240,  92.9060,  95.9400,
         98.0000, 101.0700, 102.9060, 106.4200, 107.8680, 112.4120, 114.8180,
        118.7110, 121.7600, 127.6000, 126.9040, 131.2900, 132.9050, 137.3280,
        138.9060, 140.1160, 140.9080, 144.2400, 145.0000, 150.3600, 151.9640,
        157.2500, 158.9250, 162.5000, 164.9300, 167.2600, 168.9340, 173.0400,
        174.9670, 178.4900, 180.9480, 183.8400, 186.2070, 190.2300, 192.2170,
        195.0780, 196.9670, 200.5900, 204.3830, 207.2000, 208.9800, 209.0000,
        210.0000, 222.0000, 223.0000, 226.0000, 227.0000, 232.0380, 231.0360,
        238.0290, 237.0000, 244.0000, 243.0000, 247.0000, 247.0000, 251.0000,
        252.0000, 257.0000, 258.0000, 259.0000, 262.0000, 267.0000, 268.0000,
        269.0000, 270.0000, 269.0000, 278.0000, 281.0000, 281.0000, 285.0000,
        284.0000, 289.0000, 288.0000, 293.0000, 292.0000, 294.0000, 0.0])


# -------- utils for node and edge feature in the diffusion finetuning --------

## convert dense adj to sparse with address: inaccurate with external edges
def convert_dense_adj_to_sparse_with_attr(adj, node_mask): # B x N x N
    adj = adj[node_mask]
    edge_index = (adj > 0.5).nonzero().t()
    row, col = edge_index[0], edge_index[1]
    edge_attr = adj[row, col]
    return torch.stack([row, col], dim=0), edge_attr

## estimate node and edge feature in the diffusion process
def estimate_feature_embs(target_data, source_data, prediction_model, obj='node'):
    assert obj in ['node', 'edge']
    embeds = 0
    source_x, source_edge_attr = source_data.x, source_data.edge_attr
    def mean_aggr(label, sample, label_num):
        sample = sample.to(torch.float32)
        M = torch.zeros(label_num, sample.size(0)).to(sample.device)
        M[label, torch.arange(sample.size(0))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        return torch.mm(M, sample) # for x: 119 * N * N * F => 119 * F
    if obj.lower() == 'node':
        global atom_feature_list
        feature_list = [ t.to(target_data.device) for t in atom_feature_list ]
        target_data = torch.nn.functional.one_hot(target_data.argmax(dim=-1), num_classes=target_data.size(-1)).to(torch.float32)
        embedding_list = prediction_model.graph_encoder.atom_encoder.atom_embedding_list
        atom_label = source_x[:, 0].view(-1)
        aggregated = mean_aggr(atom_label, source_x, label_num=119)
    else:
        bond_type = torch.tensor([1., 2., 3., 4., 5.]).to(target_data.device)
        target_data = torch.nn.functional.gumbel_softmax(-torch.abs(target_data.view(-1, 1)-bond_type.view(1,-1)), tau=0.01, dim=-1, hard=True)
        global bond_feature_list
        feature_list = [ t.to(target_data.device) for t in bond_feature_list ]
        embedding_list = prediction_model.graph_encoder.bond_encoder.bond_embedding_list
        bond_label = source_edge_attr[:, 0].view(-1)
        aggregated = mean_aggr(bond_label, source_edge_attr, label_num=5)
    feature_raw = torch.mm(target_data, aggregated) # (N*119/5) * (119/5*F) => N*F
    for raw_dim in range(feature_raw.size(1)):
        if raw_dim == 0:
            embeds += torch.mm(target_data, embedding_list[raw_dim].weight)
        else:
            dists = torch.abs(feature_raw[:, raw_dim].view(-1,1) - feature_list[raw_dim-1].view(1,-1))
            probs = torch.nn.functional.softmax(-dists,dim=-1)
            embeds += torch.mm(probs, embedding_list[raw_dim].weight)
    return embeds

## get raw graph feature from mol atoms and structures
def extract_graph_feature(x, adj, node_mask=None):
    adj[adj==4.0] = 1.5
    adj[adj==5.0] = 0.
    if node_mask is None:
        node_mask = (adj.sum(dim=-1)>0).bool()
    valid_num = node_mask.sum(dim=-1, keepdim=True)

    mol_rep = extract_mol_feature(x,adj,node_mask)
    graph_rep = extract_structure_feature(x, adj, node_mask)  
    final_rep = torch.concat([valid_num, mol_rep, graph_rep], dim=-1)
    adj[adj==1.5] = 4.0
    return final_rep

def extract_mol_feature(x, adj, node_mask):
    # x: (bs, n, atomic_dim)
    # adj: (bs, n, n)
    def get_stat(x, batch_index):
        size = batch_index[-1].item() + 1
        return torch.cat([scatter_mean(x, batch_index, dim=0, dim_size=size), scatter_min(x, batch_index, dim=0, dim_size=size)[0], scatter_max(x, batch_index, dim=0, dim_size=size)[0]], dim=-1)
    batch_index = node_mask.nonzero()[:,0]
    atom_index = torch.argmax(x, dim=-1)  # (bs, n)
    global atom_weight_list
    atom_weight_list = atom_weight_list.to(x.device)
    mol_weights = atom_weight_list[atom_index][node_mask]
    mol_valency = adj.sum(dim=-1)[node_mask]
    weight_feat = get_stat(mol_weights.view(-1,1), batch_index)
    val_feat = get_stat(mol_valency.view(-1,1), batch_index)
    weight_feat = weight_feat / (weight_feat.norm(dim=0, keepdim=True) + 1e-18)
    val_feat = val_feat / (val_feat.norm(dim=0, keepdim=True) + 1e-18)
    return torch.cat([weight_feat, val_feat], dim=-1)

def extract_structure_feature(x, adj, node_mask):
    batch_index = node_mask.nonzero()[:,0]
    ret_feat = scatter_add(adj.sum(dim=-1,keepdim=True)[node_mask], batch_index, dim=0, dim_size=x.size(0)) # degree
    ret_feat = torch.cat([ret_feat, scatter_add(x[node_mask], batch_index, dim=0, dim_size=x.size(0))], dim=-1)  # atom distribution
    return ret_feat


# -------- utils to convert sparse input to dense input: score model is pretrained on QM9(alterative ZINC250K): input of the augmentation --------
def split_edge_attr(edge_attr):
    enable_edge_attr_indices = (edge_attr < 4).nonzero()
    disable_edge_attr_indices = (edge_attr >= 4).nonzero()
    disable_edge_attr = edge_attr.clone()
    disable_edge_attr[enable_edge_attr_indices] = 0
    edge_attr[disable_edge_attr_indices] = 0
    return edge_attr, disable_edge_attr  

def convert_sparse_to_dense(batch_index, node_feature, edge_index, edge_attr, augment_mask=None, return_node_mask=True):
    edge_attr = edge_attr[:,0].view(-1) + 1 # index 0/1/2/3/4 to bond type S, D, T, AROMATIC and misc
    max_count = torch.unique(batch_index, return_counts=True)[1].max()
    # process edge matrix B, N, N
    enable_edge_attr, disable_edge_attr = split_edge_attr(edge_attr)
    dense_enable_adj = to_dense_adj(edge_index, batch=batch_index, edge_attr=enable_edge_attr, max_num_nodes=max_count).to(torch.float32)
    dense_disable_adj = to_dense_adj(edge_index, batch=batch_index, edge_attr=disable_edge_attr, max_num_nodes=max_count).to(torch.float32)
    if augment_mask is not None:
        dense_enable_adj = dense_enable_adj[augment_mask]
        dense_disable_adj = dense_disable_adj[augment_mask]
    # process node feature B, N, F
    atomic_num = node_feature[:,0].view(-1) + 1
    dense_atomic_num, node_mask = to_dense_batch(atomic_num, batch=batch_index, max_num_nodes=max_count) # B, N=max_cout, F=None
    if augment_mask is not None:
        dense_atomic_num = dense_atomic_num[augment_mask]
    # 6: C, 7: N, 8: O, 9: F etc.
    dense_one_hot = torch.nn.functional.one_hot(dense_atomic_num, num_classes=120).to(torch.float32)[:,:,1:] # (B N F), F = [0, 1~118, misc]
    enbale_index = torch.LongTensor([5,6,7,8])
    enable_mask = torch.zeros(119).scatter_(0, enbale_index, 1).bool()
    dense_x = dense_one_hot[:,:,enable_mask]
    dense_x_disable = dense_one_hot[:,:,~enable_mask]
    if return_node_mask:
        return dense_x, dense_x_disable, dense_enable_adj, dense_disable_adj, node_mask
    else:
        return dense_x, dense_x_disable, dense_enable_adj, dense_disable_adj

## utils for dense adj processing
def standardize_adj(adj):
    device = adj.device
    adj = (adj + adj.transpose(-1,-2)) / 2
    mask = torch.eye(adj.size(-1), adj.size(-1)).bool().unsqueeze_(0).to(device)
    adj.masked_fill_(mask, 0)
    return adj
def quantize_mol(adjs):  
    if type(adjs).__name__ == 'Tensor':
        adjs = adjs.detach().cpu()
    else:
        adjs = torch.tensor(adjs)
    torch.nan_to_num(adjs, nan=0.0)
    adjs = standardize_adj(adjs)
    adjs[adjs >= 2.5] = 3
    adjs[torch.bitwise_and(adjs >= 1.5, adjs < 2.5)] = 2
    adjs[torch.bitwise_and(adjs >= 0.5, adjs < 1.5)] = 1
    adjs[adjs < 0.5] = 0
    return adjs
def combine_graph_inputs(x, x_disable, adj, adj_disable, mode='continuous'):
    assert mode in ['continuous', 'discrete']
    x_disable = x_disable.to(x.device)
    adj_disable = adj_disable.to(adj.device)
    permutate_x_one_hot = torch.cat([torch.LongTensor([4,5,6,7,8,0,1,2,3]), torch.arange(9,119)]).to(x.device)
    x = torch.cat([x, x_disable*1.5],dim=-1)
    x = x[:, :, permutate_x_one_hot]
    if mode == 'discrete':
        adj = quantize_mol(adj)
    else:
        adj = standardize_adj(adj)
    adj = torch.maximum(adj, adj_disable)
    return x, adj


# -------- utils to convert dense input to original sparse input: output of the augmentation --------
def convert_dense_to_rawpyg(dense_x, dense_adj, augmented_labels, n_jobs=20):
    # dense_x: B, N, F; dense_adj: B, N, N; return: B, N, F, adj
    if isinstance(augmented_labels, torch.Tensor):
        augmented_labels = augmented_labels.cpu().numpy()
    dense_x = dense_x.cpu().numpy()
    dense_adj = dense_adj.cpu().numpy()
    dense_x = np.where(dense_x > 1 / dense_x.shape[-1], dense_x, 0.)
    pyg_graph_list = []
    augment_trace = np.arange(dense_x.shape[0])
    batch_split_x = np.array_split(dense_x, n_jobs)
    batch_split_adj = np.array_split(dense_adj, n_jobs)
    batch_split_labels = np.array_split(augmented_labels, n_jobs)
    batch_split_traces = np.array_split(augment_trace, n_jobs)
    with Pool(n_jobs) as pool:  # Pool created
        results = pool.map(get_pyg_data_from_dense_batch, [(batch_split_x[i], batch_split_adj[i], batch_split_labels[i], batch_split_traces[i]) for i in range(len(batch_split_adj))])
    for single_results in results:
        pyg_graph_list.extend(single_results)
    return pyg_graph_list


def get_pyg_data_from_dense_batch(params):
    batched_x, batched_adj, augmented_labels, batch_traces = params
    pyg_graph_list = []

    for b_index, (x_single, adj_single) in enumerate(zip(batched_x, batched_adj)):
        mol = construct_single_mol(x_single, adj_single)
        vcmol= valid_mol_can_with_seg(correct_mol(mol)[0], largest_connected_comp=True)[0]
        if vcmol is not None:
            graph = mol_to_graph(vcmol)
        elif vcmol is None:
            graph = mol_to_graph(mol)
        if graph is None:
            pyg_graph_list.append(int(batch_traces[b_index]))
        else:
            g = Data()
            g.__num_nodes__ = graph['num_nodes']    # ogb < 1.3.4
            g.num_nodes  = graph['num_nodes']       # ogb > 1.3.4
            g.edge_index = torch.from_numpy(graph['edge_index'])
            del graph['num_nodes']
            del graph['edge_index']
            g.y = torch.from_numpy(augmented_labels[b_index]).view(1, -1)
            if graph['edge_feat'] is not None:
                g.edge_attr = torch.from_numpy(graph['edge_feat'])
                del graph['edge_feat']
            if graph['node_feat'] is not None:
                g.x = torch.from_numpy(graph['node_feat'])
                del graph['node_feat']
            pyg_graph_list.append(g)
    return pyg_graph_list

def construct_single_mol(x, adj): # x: 9, 5; adj: 9, 9
    mol = Chem.RWMol()
    atomic_int = np.argmax(x, axis=1) + 1
    atoms_exist = atomic_int > 1 # No Hydrogen
    atomic_int = atomic_int[atoms_exist]
    for atomic_num in atomic_int:
        if atomic_num == 119:
            mol.AddAtom(Chem.Atom('*'))
        else:
            mol.AddAtom(Chem.Atom(int(atomic_num)))
    adj = adj[atoms_exist, :][:, atoms_exist]
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_simple[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                pt = Chem.GetPeriodicTable()
                if an in (7, 8, 16) and (v - pt.GetDefaultValence(an)) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol

def mol_to_graph(mol):
    # atoms
    if len(mol.GetBonds())==0 or len(mol.GetAtoms()) == 0:
        return None
    assert len(mol.GetAtoms()) > 0
    assert len(mol.GetBonds()) > 0
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)
    # bonds
    edges_list = []
    edge_features_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_feature = bond_to_feature_vector(bond)
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        edges_list.append((j, i))
        edge_features_list.append(edge_feature)
    edge_index = np.array(edges_list, dtype = np.int64).T
    edge_attr = np.array(edge_features_list, dtype = np.int64)
    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    return graph 


## utils for valid molecules check and correct
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e) # example: Explicit valence for atom # 3 C, 5, is greater than permitted
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence

def correct_mol(mol):
    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder[t])
    return mol, no_correct

def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol, sm

## functions to get atom and bond features
def atom_to_feature_vector(atom):
    atom_feature = []
    try:
        atom_prop = atom.GetAtomicNum()
    except:
        atom_prop = 'misc'
    atom_feature.append(safe_index(allowable_features['possible_atomic_num_list'], atom_prop))
    try:
        atom_prop = atom.GetChiralTag()
    except:
        atom_prop = 'misc'
    atom_feature.append(safe_index(allowable_features['possible_chirality_list'], str(atom_prop)))
    try:
        atom_prop = atom.GetTotalDegree()
    except:
        atom_prop = 'misc'
    atom_feature.append(safe_index(allowable_features['possible_degree_list'], atom_prop))
    try:
        atom_prop = atom.GetFormalCharge()
    except:
        atom_prop = 'misc'
    atom_feature.append(safe_index(allowable_features['possible_formal_charge_list'], atom_prop))
    try:
        atom_prop = atom.GetTotalNumHs()
    except:
        atom_prop = 'misc'
    atom_feature.append(safe_index(allowable_features['possible_numH_list'], atom_prop))
    try:
        atom_prop = atom.GetNumRadicalElectrons()
    except:
        atom_prop = 'misc'
    atom_feature.append(safe_index(allowable_features['possible_number_radical_e_list'], atom_prop))
    try:
        atom_prop = atom.GetHybridization()
    except:
        atom_prop = 'misc'
    atom_feature.append(safe_index(allowable_features['possible_hybridization_list'], str(atom_prop)))
    try:
        atom_prop = atom.GetIsAromatic()
    except:
        atom_prop = 'misc'
    atom_feature.append(safe_index(allowable_features['possible_is_aromatic_list'], atom_prop))
    try:
        atom_prop = atom.IsInRing()
    except:
        atom_prop = 'misc'
    atom_feature.append(safe_index(allowable_features['possible_is_in_ring_list'], atom_prop))
    return atom_feature

def bond_to_feature_vector(bond):
    bond_feature = []
    try:
        bond_prop = bond.GetBondType()
    except:
        bond_prop = 'misc'
    bond_feature.append(safe_index(allowable_features['possible_bond_type_list'], str(bond_prop)))
    try:
        bond_prop = bond.GetStereo()
    except:
        bond_prop = 'misc'
    bond_feature.append(safe_index(allowable_features['possible_bond_stereo_list'], str(bond_prop)))
    try:
        bond_prop = bond.GetIsConjugated()
    except:
        bond_prop = 'misc'
    bond_feature.append(safe_index(allowable_features['possible_is_conjugated_list'], bond_prop))
    return bond_feature