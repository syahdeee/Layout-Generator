import dgl
import math
import torch
import numpy as np
import torch.nn as nn
from scipy import sparse as sp

def construct_dgl_graph(objects, triples, pos_enc_dim=16, triplet_type=None):
    s, p, o = triples.chunk(3, dim=1)
    p = p.squeeze(1)
    s, o = s.squeeze(), o.squeeze()
    s = s.view(-1) 
    o = o.view(-1)  
    g = dgl.graph((s, o))
    pad_size = len(objects) - g.num_nodes()
    g.add_nodes(pad_size)
    g.ndata['feat'] = objects
    g.edata['feat'] = p
    if triplet_type is not None:
        g.edata['type'] = triplet_type
    laplacian_positional_encoding(g, pos_enc_dim)

    return g

class SinePositionalEncoding(nn.Module):

    def __init__(self, emb_size, dropout=0.1, max_len=10):
        super(SinePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb_size = emb_size

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.emb_size)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
        DGL implementation
    """
    n_nodes = g.number_of_nodes()
    pad_size = pos_enc_dim + 1 - n_nodes
    pad_size = max(0, pad_size)

    # Laplacian
    A = g.adjacency_matrix().to_dense().numpy().astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1)
                 ** -0.5, dtype=float).toarray()
    L = np.eye(n_nodes) - N @ A @ N
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort() 
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    EigVec = np.pad(EigVec, (0, pad_size))[:n_nodes]
    g.ndata['lap_pos_enc'] = torch.from_numpy(
        EigVec[:, 1:pos_enc_dim+1]).float()

    return g

def dgl_coco_collate_all(dataset, pos_enc_dim=16):
    all_graphs, all_boxes = [], [], []
    max_objs = 0
    
    for sample in dataset:
        objs, boxes, triples = sample
        if objs.dim() == 0 or triples.dim() == 0:
            continue
        O, T = objs.size(0), triples.size(0)

        max_objs = max(max_objs, O)

        pad_size = max_objs - O
        boxes = torch.cat((boxes, torch.LongTensor([[0, 0, 1, 1]] * pad_size)))
        all_boxes.append(boxes)
        objs = torch.cat((objs, torch.LongTensor([0] * pad_size)))

        triples = triples.clone()
        graph = construct_dgl_graph(objs, triples, pos_enc_dim)
        all_graphs.append(graph)
        
    all_boxes = torch.cat(all_boxes)
    all_graphs = dgl.batch(all_graphs)
    out = (all_graphs, all_boxes)
    return out