import os
import torch
import logging
import pandas as pd
import os.path as osp
from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split

from .data_utils import read_graph_list
logger = logging.getLogger(__name__)

class PolymerRegDataset(InMemoryDataset):
    def __init__(self, name='plym-oxygen', root ='data', transform=None, pre_transform = None):
        '''
            - name (str): name of the dataset: plym-oxygen/melting/glass/density
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        ''' 
        self.name = name
        self.dir_name = '_'.join(name.split('-'))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        self.processed_root = osp.join(osp.abspath(self.root))

        self.num_tasks = 1
        self.eval_metric = 'rmse'
        self.task_type = 'regression'
        self.__num_classes__ = '-1'
        self.binary = 'False'

        super(PolymerRegDataset, self).__init__(self.processed_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.total_data_len = self.data.y.size(0)

    def get_idx_split(self, split_type = 'random'):
        if split_type is None:
            split_type = 'random'
        path = osp.join(self.root, 'split', split_type)
        if not os.path.exists(path):
            os.makedirs(path)

        try: 
            train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
            valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
            test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]
        except:
            print('Splitting with random seed 42 and ratio 0.6/0.1/0.3')
            full_idx = list(range(self.total_data_len))
            train_ratio, valid_ratio, test_ratio = 0.6, 0.1, 0.3
            train_idx, test_idx, _, _ = train_test_split(full_idx, full_idx, test_size=test_ratio, random_state=42)
            train_idx, valid_idx, _, _ = train_test_split(train_idx, train_idx, test_size=valid_ratio/(valid_ratio+train_ratio), random_state=42)
            df_train = pd.DataFrame({'train': train_idx})
            df_valid = pd.DataFrame({'valid': valid_idx})
            df_test = pd.DataFrame({'test': test_idx})
            df_train.to_csv(osp.join(path, 'train.csv.gz'), index=False, header=False, compression="gzip")
            df_valid.to_csv(osp.join(path, 'valid.csv.gz'), index=False, header=False, compression="gzip")
            df_test.to_csv(osp.join(path, 'test.csv.gz'), index=False, header=False, compression="gzip")
        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def process(self):
        print('begin processing polymer data at folder: ' , osp.join(self.root, 'raw' ,self.name.split('-')[1]+'_raw.csv'))
        data_list = read_graph_list(osp.join(self.root, 'raw' ,self.name.split('-')[1]+'_raw.csv'), property_name=self.name, process_labeled=True)
        # print(data_list[:3])
        self.total_data_len = len(data_list)
        print('Labeled Finished with length ', self.total_data_len)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
