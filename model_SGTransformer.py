import dgl
import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from training import get_lap_pos_enc

def build_mlp(dim_list, activation='gelu', batch_norm='none',
              dropout=0.1, final_nonlinearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
            
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        
        if not final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
                
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
            
    return nn.Sequential(*layers)

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func



"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(fn.u_dot_v('K_h', 'Q_h', 'score'))

        # Scaling
        g.apply_edges(lambda edges: {'score': edges.data['score'] / torch.sqrt(torch.tensor(self.out_dim))})

        # Use available edge features to modify the scores (proj_e)
        g.apply_edges(lambda edges: {'score': edges.data['score'] + edges.data['proj_e']})

        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(lambda edges: {'e_out': edges.data['score']})

        # Softmax
        g.edata['score'] = dgl.ops.edge_softmax(g, g.edata['score'])

        # Message passing
        g.update_all(fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))

        return g

    def forward(self, g, h, e):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        h_out = g.ndata['wV']
        e_out = g.edata['e_out']
        
        return h_out, e_out

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False, mixed_precision=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.mixed_precision = mixed_precision

        self.attention = MultiHeadAttentionLayer(
            in_dim, out_dim//num_heads, num_heads, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)

        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, e, amp=False):
        h_in1 = h  # for first residual connection
        e_in1 = e  # for first residual connection

        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e)

        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h  # residual connection
            e = e_in1 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h  # for second residual connection
        e_in2 = e  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h  # residual connection
            e = e_in2 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads, self.residual)
class GraphTransformerNet(nn.Module):
    def __init__(self,
                 n_objs ,
                 n_rels ,
                 emb_size,
                 n_heads,
                 n_enc_layers,
                 pos_enc_dim,
                 dropout=0.1,
                 layer_norm=False,
                 batch_norm=True,
                 residual=True,
                 skip_edge_feat=False,
                 mixed_precision=False,
                 lap_pos_enc=True):
        super().__init__()
        out_dim = emb_size

        if lap_pos_enc:
            pos_enc_dim = pos_enc_dim
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, emb_size)

        self.lap_pos_enc = lap_pos_enc
        self.skip_edge_feat = skip_edge_feat
        self.embedding_h = nn.Embedding(n_objs, emb_size)

        if not skip_edge_feat:
            self.embedding_e = nn.Embedding(n_rels, emb_size)
        else:
            self.embedding_e = nn.Linear(1, emb_size)

        self.in_feat_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(emb_size, emb_size, n_heads, dropout, layer_norm,
                                  batch_norm, residual, mixed_precision=mixed_precision)
            for _ in range(n_enc_layers-1)])
        self.layers.append(GraphTransformerLayer(emb_size, out_dim, n_heads, dropout,
                           layer_norm, batch_norm, residual, mixed_precision=mixed_precision))
        self.box_regressor = nn.Sequential(
            build_mlp([emb_size, emb_size //
                      2, emb_size // 4]),
            nn.Linear(emb_size // 4, 4),
            nn.Sigmoid()
        )
        
        self.box_regressor.apply(_init_weights)
    def forward(self, g):

        h = g.ndata['feat']
        e = g.edata['feat']
        h_lap_pos_enc = get_lap_pos_enc(g)
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.skip_edge_feat:
            e = torch.ones(e.size(0), 1, device=h.device)
        e = self.embedding_e(e)

        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        boxes = self.box_regressor(h)
        return boxes

    def loss(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss
    
def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)


