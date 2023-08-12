import torch
import math
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.kge import KGEModel
import torch
from torch_geometric.nn import RGCNConv, GATConv

def encode_features(x, model):
    x_files, x_processes, x_sockets = 0, 0, 0
    files = x[(x[:, 0] == 0).nonzero().squeeze(1)]
    for h in [0, 1, 2, 3, 4]:
        x_files += model.embeddings[h](files[:, h])
    processes = x[(x[:, 0] == 1).nonzero().squeeze(1)]
    for h in [0, 5]:
        x_processes += model.embeddings[h](processes[:, h])
    sockets = x[(x[:, 0] == 2).nonzero().squeeze(1)]
    for h in [0, 6, 7, 8, 9]:
        x_sockets += model.embeddings[h](sockets[:, h])

    x_new = torch.zeros((x.shape[0], x_sockets.shape[-1])).to(x.device)
    x_new[(x[:, 0] == 0).nonzero().squeeze(1)] = x_files
    x_new[(x[:, 0] == 1).nonzero().squeeze(1)] = x_processes
    x_new[(x[:, 0] == 2).nonzero().squeeze(1)] = x_sockets
    return x_new


class BaselineEdgePredictor(torch.nn.Module):

    def __init__(self, node_dim=3, edge_dim=27, include_edge=False, include_features=True, node_embedding_dim=[()]):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.include_edge = include_edge
        self.include_features = include_features
        embed_dim = node_embedding_dim[0][1]
        if not include_features:
            node_embedding_dim = [node_embedding_dim[0]]
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(c, dim) for (c, dim) in node_embedding_dim])

        self.src_fc = torch.nn.Embedding(3, embed_dim)
        self.dst_fc = torch.nn.Embedding(3, embed_dim)
        self.edge = torch.nn.Linear(edge_dim, embed_dim)
        self.out_fc = torch.nn.Linear(embed_dim, 1)
        self.num_gnn_layers = 0
    
    def update(self, src, pos_dst, t, msg, static_x):
        return 0
    
    def detach_memory(self):
        return 0

    def reset_memory(self):
        return 0
    
    def forward(self, batch, n_id, msg, t, edge_index, id_mapper, src_indic):
        if self.include_features:
            h_src = encode_features(batch.x[batch.src], self)
            h_pos_dst = encode_features(batch.x[batch.dst], self)
            h_neg_dst = encode_features(batch.x[batch.neg_dst], self)
        else:
            h_src = batch.x[batch.src, 0 ]
            h_pos_dst = batch.x[batch.dst, 0]
            h_neg_dst = batch.x[batch.neg_dst, 0]
            h_src = self.src_fc(h_src)
            h_pos_dst = self.dst_fc(h_pos_dst)
            h_neg_dst = self.dst_fc(h_neg_dst)
        target_msg =  self.edge(batch['msg'].squeeze().to(msg.device))
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src + h_neg_dst)

        return self.out_fc(h_pos_edge + target_msg), self.out_fc(h_neg_edge + target_msg)
    

class EdgeBank(torch.nn.Module):
    def __init__(self, node_dim=3, edge_dim=27, include_edge=False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.include_edge = include_edge
        if include_edge:
            self.cube = torch.nn.parameter.Parameter(torch.zeros(node_dim, node_dim, edge_dim), requires_grad=False)
        else:
            self.cube = torch.nn.parameter.Parameter(torch.zeros(node_dim, node_dim, 1), requires_grad=False)
        self.num_gnn_layers = 0
    
    def update(self, src, pos_dst, t, msg, static_x):
        return 0
    
    def detach_memory(self):
        return 0

    def reset_memory(self):
        return 0

    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        h_src = batch.x[batch.src]
        h_pos_dst = batch.x[batch.dst]

        i = h_src.argmax(1).cpu()
        j = h_pos_dst.argmax(1).cpu()
        if self.include_edge:
            k = batch.msg.argmax(1).cpu()
        else:
            k = 0
        if self.train:
            self.cube[i, j, k] = 1

        pos = self.cube[i, j, k]

        h_neg_dst = batch.x[batch.neg_dst]
        j = h_neg_dst.argmax(1).cpu()
        neg = self.cube[i, j, k]
        
        return pos.unsqueeze(1), neg.unsqueeze(1)

class GNNEdgePredictor(torch.nn.Module):
    def __init__(self, node_dim, embed_dim):
        super().__init__()

        self.src_fc = torch.nn.Linear(node_dim, embed_dim)
        self.dst_fc = torch.nn.Linear(node_dim, embed_dim)
        self.out_fc = torch.nn.Linear(embed_dim, 1)
    
        self.num_gnn_layers = 0
    
    def update(self, src, pos_dst, t, msg, static_x):
        return 0
    
    def detach_memory(self):
        return 0

    def reset_memory(self):
        return 0

    def forward(self, h_src, h_dst, msg):
        h_src = self.src_fc(h_src)
        h_dst = self.dst_fc(h_dst)
        h_edge = torch.nn.functional.relu(torch.cat((h_src, h_src)) + h_dst)
        h_edge = h_edge + msg
        out = self.out_fc(h_edge).squeeze()
        return out[:len(out)//2], out[len(out)//2:]


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, node_dim, node_embedding_dim,
                 num_relations, dropout=False, num_layers=2, include_features=True, include_edge=False):
        super().__init__()
        self.dropout = dropout
        self.embed_edge = torch.nn.Embedding(num_relations, in_channels)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, edge_dim=in_channels)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, edge_dim=in_channels)
        self.include_features = include_features
        if not include_features:
            node_embedding_dim = [node_embedding_dim[0]]
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(c, dim) for (c, dim) in node_embedding_dim])
        self.link_predictor = GNNEdgePredictor(out_channels, out_channels)
        self.num_gnn_layers = num_layers
        self.include_edge = include_edge
    
    def update(self, src, pos_dst, t, msg, static_x):
        return 0
    
    def detach_memory(self):
        return 0

    def reset_memory(self):
        return 0

    def forward(self, batch, src, pos_dst, neg_dst, msg, x, edge_index):
        x = encode_features(x, self)
        edge_attr = self.embed_edge(msg.squeeze().abs().long())
        if self.dropout:
            x = F.dropout(x, p=0.6, training=self.training)
        x = torch.nn.functional.relu(self.conv1(x, edge_index.long(), edge_attr=edge_attr))
        if self.dropout:
            x = F.dropout(x, p=0.6, training=self.training)
        x = torch.nn.functional.relu(self.conv2(x, edge_index.long(), edge_attr=edge_attr))
        h = x
        h_src = h[src]
        h_dst = h[torch.cat((pos_dst, neg_dst))]
        if self.include_edge:
            target_msg =  self.embed_edge(torch.cat((batch['msg'].squeeze().to(x.device), batch['msg'].squeeze().to(x.device))))
        else:
            target_msg = 0
        pos_out, neg_out = self.link_predictor(h_src, h_dst, target_msg)
        return pos_out.unsqueeze(1), neg_out.unsqueeze(1)
    

class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, node_dim, node_embedding_dim,
                 num_relations, num_layers=2, dropout=False, include_features=True, include_edge=False):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels* 4, num_relations)
        self.conv2 = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.conv2.append(RGCNConv(hidden_channels* 4, hidden_channels* 4, num_relations))
        
        self.include_features = include_features
        if not include_features:
            node_embedding_dim = [node_embedding_dim[0]]
        self.embed_edge = torch.nn.Embedding(num_relations, hidden_channels* 4)
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(c, dim) for (c, dim) in node_embedding_dim])
        self.link_predictor = GNNEdgePredictor(hidden_channels* 4, hidden_channels*4)
        self.num_gnn_layers = num_layers
        self.include_edge = include_edge
    
    def update(self, src, pos_dst, t, msg, static_x):
        return 0
    
    def detach_memory(self):
        return 0

    def reset_memory(self):
        return 0
    
    def forward(self, batch, src, pos_dst, neg_dst, msg, x, edge_index):
        x = encode_features(x, self)
        x = self.conv1(x, edge_index.long(), msg.squeeze()).relu()
        for layer in self.conv2:
            x = layer(x, edge_index.long(), msg.squeeze()).relu()
        h_src = x[src]
        h_dst = x[torch.cat((pos_dst, neg_dst))]
        if self.include_edge:
            target_msg =  self.embed_edge(torch.cat((batch['msg'].squeeze().to(x.device), batch['msg'].squeeze().to(x.device))))
        else:
            target_msg = 0
        pos_out, neg_out = self.link_predictor(h_src, h_dst, target_msg)
        return pos_out.unsqueeze(1), neg_out.unsqueeze(1)
    
class TransR(KGEModel):
    r"""The TransR model 
    .. note::

        For an example of using the :class:`TransE` model, see
        `examples/kge_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        kge_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (int, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        p_norm (int, optional): The order embedding and distance normalization.
            (default: :obj:`1.0`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        p_norm: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        self.p_norm = p_norm
        self.margin = margin
        self.num_relations = num_relations
        self.rel_project = torch.nn.Embedding(num_relations, hidden_channels * hidden_channels)
    

        self.reset_parameters()

    def reset_parameters(self):
        identity = torch.eye(self.hidden_channels).flatten().repeat(self.num_relations, 1)
        self.rel_project.weight.data = identity
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1,
                    out=self.rel_emb.weight.data)


    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)

        head = F.normalize(head, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)

        proj_rel = self.rel_project(rel_type).reshape(-1, self.hidden_channels, self.hidden_channels)
        head = (head.unsqueeze(1) @ proj_rel).squeeze(1)
        tail = (tail.unsqueeze(1) @ proj_rel).squeeze(1)

        head = F.normalize(head, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)

        # Calculate *negative* norm:
        return -((head + rel) - tail).norm(p=self.p_norm, dim=-1)

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )