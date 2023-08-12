import os.path as osp
from typing import Callable, Optional
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, TemporalData, HeteroData, Data
from torch_geometric.loader import LinkNeighborLoader
import os
import torch_geometric.transforms as T
from torch.utils.data import Dataset
import pandas as pd
import ast

def f(x, i):
    return x[i]

class DARPADataset_Temporal(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'temporal_processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    @property
    def metadata(self) -> tuple:
        return self.data_metadata

    def process(self):
        import pandas as pd
        path = osp.join(self.root, self.name)
        df = pd.read_csv(os.path.join(path, "edges.csv"))
        src = torch.from_numpy(df["src"].values).to(torch.long)
        dst = torch.from_numpy(df["dst"].values).to(torch.long)
        t = torch.from_numpy(df["time"].values).to(torch.long)
        # DARPA TRACE and THEIA have up to 27 different system calls
        msg = torch.nn.functional.one_hot(torch.from_numpy(df["syscall"].values).long(), 27).float()
        nodes = pd.read_csv(os.path.join(path, "attributed_nodes.csv"), index_col=0)
        nodes.loc[~nodes.port_class.isna(), 'port_class'] = nodes.loc[~nodes.port_class.isna(), 'port_class'].apply(lambda x: ast.literal_eval(x))
        x = torch.zeros((len(nodes), 10)).long()
        
        # Heterogeneous metadata
        df['src_type'] = nodes.type.loc[src].values
        df['dst_type'] = nodes.type.loc[dst].values
        combinations = df.drop_duplicates(['src_type', 'dst_type', 'syscall'])[['src_type', 'dst_type', 'syscall']]
        combinations["src_type"].replace([0, 1, 2], ['file', 'process', 'socket'], inplace=True)
        combinations["dst_type"].replace([0, 1, 2], ['file', 'process', 'socket'], inplace=True)
        self.data_metadata = (['file', 'process', 'socket'], [])
        for _, row in combinations.iterrows():
            self.data_metadata[1].append((row.src_type, 'rel_'+str(row.syscall), row.dst_type))
            self.data_metadata[1].append((row.dst_type, 'rev_rel_'+str(row.syscall), row.src_type))

        # Files (node_type, root, extension_1, extension_2, extension_3)
        n_files = len(nodes[nodes['type'] == 0])
        files = nodes[:n_files]
        files['extensions_class'] = files['extensions_class'].apply(lambda x: ast.literal_eval(x))
        f1 = torch.from_numpy((files[files.extensions_class.map(len)>0].extensions_class.map(lambda x: f(x, 0)) + 1).values)
        f2 = torch.zeros_like(f1)
        f2[(files.extensions_class.map(len)>1).values] = torch.from_numpy((files[files.extensions_class.map(len)>1].extensions_class.map(lambda x: f(x, 1)) + 1).values) 
        f3 = torch.zeros_like(f1)
        f3[(files.extensions_class.map(len)>2).values] = torch.from_numpy((files[files.extensions_class.map(len)>2].extensions_class.map(lambda x: f(x, 2)) + 1).values)

        x[:n_files, [0, 1, 2, 3, 4]] = torch.stack((torch.zeros(n_files).long(),
                                                  torch.from_numpy(files.root_class.values).long(),
                                                  f1.long(),
                                                  f2.long(),
                                                  f3.long())
                                                  ,dim=1)
        
        # Processes (node_type, processs class)
        n_processses = len(nodes[nodes['type'] == 1])
        x[n_files:n_files + n_processses, [0, 5]] = torch.stack((torch.ones(n_processses).long(),
                                                                 torch.from_numpy(nodes.processes_class.values[n_files:n_files+n_processses]).long()),
                                                                 dim=1)
        
        # Sockets (node_type, private, ip_class)
        n_sockets = len(nodes[nodes['type'] == 2])

        sockets = nodes.iloc[n_files + n_processses:n_sockets + n_files + n_processses]
        s1 = torch.from_numpy((sockets[sockets.port_class.map(len)>0].port_class.map(lambda x: f(x, 0)) + 1).values)
        s2 = torch.zeros_like(s1)
        s2[(sockets.port_class.map(len)>1).values] = torch.from_numpy((sockets[sockets.port_class.map(len)>1].port_class.map(lambda x: f(x, 1)) + 1).values) 
        s3 = torch.zeros_like(s1)
        s3[(sockets.port_class.map(len)>2).values] = torch.from_numpy((sockets[sockets.port_class.map(len)>2].port_class.map(lambda x: f(x, 2)) + 1).values)

        x[n_files + n_processses:n_sockets + n_files + n_processses, [0, 6, 7, 8, 9]] = torch.stack((torch.ones(n_sockets).long()*2,
                                                                                               torch.from_numpy(sockets.private.values).long(),
                                                                                               s1.long(),
                                                                                               s2.long(),
                                                                                               s3.long()),
                                                                                               dim=1)



        ext_roll = torch.from_numpy(df.ext_roll.values)
        hash_id = torch.from_numpy(df.hash_id.values)
        malicious = torch.from_numpy(df.malicious.values)
        

        data = TemporalData(src=src, dst=dst, t=t, msg=msg, x=x, ext_roll=ext_roll, hash_id=hash_id, malicious=malicious)
        data.metadata = self.data_metadata
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
    

# Not used
class DARPADataset_HeteroStatic(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'heterostatic_processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'


    def process(self):
        import pandas as pd
        path = osp.join(self.root, self.name)
        df = pd.read_csv(os.path.join(path, "edges.csv"))
        edge_types = torch.load(os.path.join(path, "edge_features.pt"), map_location=torch.device('cpu')).to(torch.float).argmax(1).numpy()
        node_types = torch.load(os.path.join(path, "node_features.pt"), map_location=torch.device('cpu')).to(torch.float).argmax(1).unsqueeze(1)
        df['edge_type'] = edge_types
        # TransR embeddings
        embeddings = torch.load(os.path.join(path, "model.pt"), map_location=torch.device('cpu'))
        node_emb = embeddings['model_state_dict']['node_emb.weight'].cpu()
        rel_emb = embeddings['model_state_dict']['rel_emb.weight'].cpu()
        projections = embeddings['model_state_dict']['rel_project.weight'].cpu()
        
        
        # dataset split and create heterogeneous data
        # Add nodes
        data_splits = []
        df = df[["src", "dst", "ext_roll", "edge_type", "hash_id", "malicious"]]
        df = pd.concat([df[df["ext_roll"] == 0].drop_duplicates(), df[df["ext_roll"] > 0]])
        df = df.reset_index()
        train_len = len(df[df['ext_roll'] == 0])
        disjoint_train_slice = np.random.choice(range(0, train_len), int(0.7*train_len), False)
        df.loc[disjoint_train_slice, 'ext_roll'] = -1
        for i in range(0, 3):
            previous = df[df['ext_roll'] < i]
            current = df[df['ext_roll'] == i]
            node_ids = list(set(df['src']).union(set(df['dst'])))
            data = HeteroData()

            data['node'].x = node_emb
            data['node'].node_class = node_types
            data['node'].node_ids = node_ids

            # Add edges for each edge_type
            for rel in df['edge_type'].unique():
                filtered_previous = previous[previous['edge_type'] == rel]
                filtered_current = current[current['edge_type'] == rel]
                data["node", 'rel_' + str(rel), "node"].edge_index = torch.stack((torch.Tensor(filtered_previous.src.values).int(), torch.Tensor(filtered_previous.dst.values).int()))
                data["node", 'rel_' + str(rel), "node"].rel_emb = rel_emb[filtered_previous['edge_type'].values]
                data["node", 'rel_' + str(rel), "node"].edge_class = torch.Tensor(filtered_previous['edge_type'].values)
                data["node", 'rel_' + str(rel), "node"].hash_id = filtered_current['hash_id'].values
                data["node", 'rel_' + str(rel), "node"].edge_label_index = torch.stack((torch.Tensor(filtered_current.src.values).int(), torch.Tensor(filtered_current.dst.values).int()))
                
            data = T.ToUndirected('mean', False)(data)
            data['node'].projections = projections
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_splits.append(data)
        torch.save(self.collate(data_splits), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
    

class DARPADataset_Static(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'static_processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self):
        return self[0].x.shape[0]
    
    @property
    def msg(self):
        return self[0].edge_attr
    
    @property
    def x(self):
        return self[0].x
    
    def process(self):
        import pandas as pd
        path = osp.join(self.root, self.name)
        df = pd.read_csv(os.path.join(path, "edges.csv"))
        df['edge_type'] = df["syscall"]
        
        # Add nodes
        df = df[["src", "dst", "ext_roll", "edge_type", "hash_id", "malicious"]]
        df = pd.concat([df[df["ext_roll"] == 0].drop_duplicates(subset=["src", "dst", "ext_roll", "edge_type"]), df[df["ext_roll"] > 0]])
        df = df.reset_index()
        train_len = len(df[df['ext_roll'] == 0])
        disjoint_train_slice = np.random.choice(range(0, train_len), int(0.7*train_len), False)
        df.loc[disjoint_train_slice, 'ext_roll'] = -1
        dataset = []

        nodes = pd.read_csv(os.path.join(path, "attributed_nodes.csv"), index_col=0)
        nodes.loc[~nodes.port_class.isna(), 'port_class'] = nodes.loc[~nodes.port_class.isna(), 'port_class'].apply(lambda x: ast.literal_eval(x))
        x = torch.zeros((len(nodes), 10)).long()

        # Files (node_type, root, extension_1, extension_2, extension_3)
        n_files = len(nodes[nodes['type'] == 0])
        files = nodes[:n_files]
        files['extensions_class'] = files['extensions_class'].apply(lambda x: ast.literal_eval(x))
        f1 = torch.from_numpy((files[files.extensions_class.map(len)>0].extensions_class.map(lambda x: f(x, 0)) + 1).values)
        f2 = torch.zeros_like(f1)
        f2[(files.extensions_class.map(len)>1).values] = torch.from_numpy((files[files.extensions_class.map(len)>1].extensions_class.map(lambda x: f(x, 1)) + 1).values) 
        f3 = torch.zeros_like(f1)
        f3[(files.extensions_class.map(len)>2).values] = torch.from_numpy((files[files.extensions_class.map(len)>2].extensions_class.map(lambda x: f(x, 2)) + 1).values)

        x[:n_files, [0, 1, 2, 3, 4]] = torch.stack((torch.zeros(n_files).long(),
                                                  torch.from_numpy(files.root_class.values).long(),
                                                  f1.long(),
                                                  f2.long(),
                                                  f3.long())
                                                  ,dim=1)
        
        # Processes (node_type, processs class)
        n_processses = len(nodes[nodes['type'] == 1])
        x[n_files:n_files + n_processses, [0, 5]] = torch.stack((torch.ones(n_processses).long(),
                                                                 torch.from_numpy(nodes.processes_class.values[n_files:n_files+n_processses]).long()),
                                                                 dim=1)
        
        # Sockets (node_type, private, ip_class)
        n_sockets = len(nodes[nodes['type'] == 2])

        sockets = nodes.iloc[n_files + n_processses:n_sockets + n_files + n_processses]
        s1 = torch.from_numpy((sockets[sockets.port_class.map(len)>0].port_class.map(lambda x: f(x, 0)) + 1).values)
        s2 = torch.zeros_like(s1)
        s2[(sockets.port_class.map(len)>1).values] = torch.from_numpy((sockets[sockets.port_class.map(len)>1].port_class.map(lambda x: f(x, 1)) + 1).values) 
        s3 = torch.zeros_like(s1)
        s3[(sockets.port_class.map(len)>2).values] = torch.from_numpy((sockets[sockets.port_class.map(len)>2].port_class.map(lambda x: f(x, 2)) + 1).values)

        x[n_files + n_processses:n_sockets + n_files + n_processses, [0, 6, 7, 8, 9]] = torch.stack((torch.ones(n_sockets).long()*2,
                                                                                               torch.from_numpy(sockets.private.values).long(),
                                                                                               s1.long(),
                                                                                               s2.long(),
                                                                                               s3.long()),
                                                                                               dim=1)
        for i in range(-1, 3):
            mask = df['ext_roll'].values == i
            data = Data()
            data.x = x

            # Add edges
            data.edge_index = torch.stack((torch.Tensor(df.src.values).int(), torch.Tensor(df.dst.values).int()))[:, mask]
            data.edge_attr = torch.tensor(df['edge_type'].values)[mask].unsqueeze(1)
            data.hash_id = torch.tensor(df.hash_id.values)[mask].unsqueeze(1)
            data.malicious = torch.tensor(df.malicious.values)[mask].unsqueeze(1)
            dataset.append(data)
        
        torch.save(self.collate(dataset), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
    
# Not used
class DARPADataset_TransR(Dataset):
    def __init__(self, path, name, mode):
        path = os.path.join(path, name)
        self.df = pd.read_csv(os.path.join(path, "edges.csv"))
        if mode == "train":
            slice = self.df[self.df['ext_roll'] == 0].index
        elif mode == "val":
            slice = self.df[self.df['ext_roll'] == 1].index
        elif mode == "test":
            slice = self.df[self.df['ext_roll'] == 2].index
        elif mode == "all":
            slice = range(0, len(self.df))
        else:
            raise NotImplementedError
        self.src = torch.from_numpy(self.df["src"].values).to(torch.long)[slice]
        self.dst = torch.from_numpy(self.df["dst"].values).to(torch.long)[slice]
        self.msg = torch.load(os.path.join(path, "edge_features.pt")).to(torch.float).argmax(1).unsqueeze(1)[slice]
        self.x = torch.load(os.path.join(path, "node_features.pt")).to(torch.float).argmax(1).unsqueeze(1)
        self.num_nodes = self.x.shape[0]
    
    def tot_len(self):
        return len(self.df)

    def tot_num_nodes(self):
        return len(self.feats)
    
    def __len__(self):
        return len(self.msg)

    def __getitem__(self, idx):
        return self.src[idx], self.dst[idx], self.msg[idx]


# Not used
def hetero_graph_to_homogeneous_loaders(data, device, batch_size=200, num_neighbors=[10, 10]):
    new_data = []
    loaders = []
    for i in range(3):
        edge_types = data[0].edge_types
        edge_label_index = []
        temp = data[i]
        for triplet in edge_types:
            if 'rev' in triplet[1]:
                temp[triplet].edge_class = -data[i][triplet].edge_class
                continue
            edge_label_index.append(temp[triplet].edge_label_index)
        new_data.append(temp.to_homogeneous())
        new_data[i].edge_label_index = torch.cat(edge_label_index, 1)
        loader = LinkNeighborLoader(
            data = new_data[i].to(device),
            num_neighbors=num_neighbors,
            edge_label_index=new_data[i].edge_label_index,
            edge_label=None,
            batch_size=batch_size,
            shuffle=True,
            neg_sampling="triplet"
        )
        loaders.append(loader) 
    return loaders


