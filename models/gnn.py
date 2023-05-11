import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .conv import GNN_node, GNN_node_Virtualnode


class GNN(torch.nn.Module):
    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, gnn_type = 'gin', drop_ratio = 0.5, graph_pooling = "max", norm_layer = 'batch_norm'):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        ### GNN to generate node embeddings
        gnn_name = gnn_type.split('-')[0]
        if 'virtual' in gnn_type:
            self.graph_encoder = GNN_node_Virtualnode(num_layer, emb_dim, JK = 'last', drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name, norm_layer = norm_layer)
        else:
            self.graph_encoder = GNN_node(num_layer, emb_dim, JK = 'last', drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name, norm_layer = norm_layer)
        ### Poolinwg function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        rep_dim = emb_dim
        self.predictor = torch.nn.Sequential(torch.nn.Linear(rep_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Dropout(), torch.nn.Linear(2*emb_dim, self.num_tasks))


    def forward(self, batched_data, encode_raw = True):
        h_node, _ = self.graph_encoder(batched_data, encode_raw)
        h_graph = self.pool(h_node, batched_data.batch)
        return self.predictor(h_graph), h_graph
