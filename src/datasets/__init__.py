import torch

from torch_geometric.datasets import JODIEDataset
from numpy.random import default_rng
import numpy
from .darpa import DARPADataset_Temporal, DARPADataset_HeteroStatic, DARPADataset_TransR, DARPADataset_Static


JODIE = ['Wikipedia', "Reddit", "MOOC", "LastFM"]
REC_SYS = [] # TODO
DARPA = ["darpa_trace_0to125", "darpa_trace_126to210", "darpa_trace_116to125", "darpa_trace_0to210", "darpa_theia_0to25"]
DATA_NAMES = JODIE + REC_SYS + DARPA

    
def get_dataset(root, name, version, seed, metadata=False):
    rng = default_rng(seed)
    data_metadata = ()
    if name in JODIE:
        dataset = JODIEDataset(root, name.lower())
        data = dataset[0]
        data.x = torch.tensor(rng.random((data.num_nodes,1), dtype=numpy.float32))
    elif name in DARPA:
        if version == 'temporal':
            dataset = DARPADataset_Temporal(root, name)
            data = dataset[0]
            data_metadata = data.metadata
            del data.metadata
        elif version == 'heterostatic':
            data = DARPADataset_HeteroStatic(root, name)
        elif version == 'static':
            dataset = DARPADataset_Static(root, name)
            data = dataset
        elif version == 'transR':
            data = [DARPADataset_TransR(root, name, mode) for mode in ['train', 'val', 'test']]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    return data, data_metadata