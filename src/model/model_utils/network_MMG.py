import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model_utils.network_util import (MLP, Aggre_Index, Gen_Index,
                                                build_mlp)
from src.model.transformer.attention import MultiHeadAttention


class DistanceAwareMasking(nn.Module):
    def __init__(self, dim_coord, dim_out):
        super().__init__()
        self.input_dim = dim_coord + 1  # 3 + 1 = 4
        self.gd = build_mlp([self.input_dim, dim_coord, dim_out], do_bn=False, on_last=False)
    
    def forward(self, node_coords):
        num_nodes = node_coords.size(0)
        
        node_i = node_coords.unsqueeze(1).repeat(1, num_nodes, 1)
        node_j = node_coords.unsqueeze(0).repeat(num_nodes, 1, 1)
        
        diff = node_i - node_j  # [num_nodes, num_nodes, 3]
        dist = torch.norm(diff, dim=2, keepdim=True)  # [num_nodes, num_nodes, 1]
        
        features = torch.cat([diff, dist], dim=2)  # [num_nodes, num_nodes, 4]
        
        mask = self.gd(features.view(-1, self.input_dim)).view(num_nodes, num_nodes, -1)
        
        return mask


class DistanceAwareMultiHeadAttention(nn.Module):
    def __init__(self, dim_node, num_heads, dim_head):
        super().__init__()
        self.dim_node = dim_node
        self.num_heads = num_heads
        self.dim_head = dim_head
        
        self.q_proj = nn.Linear(dim_node, dim_head * num_heads)
        self.k_proj = nn.Linear(dim_node, dim_head * num_heads)
        self.v_proj = nn.Linear(dim_node, dim_head * num_heads)
        self.out_proj = nn.Linear(dim_head * num_heads, dim_node)
        
        self.distance_masking = DistanceAwareMasking(3, num_heads)
    
    def forward(self, x, node_coords):
        batch_size = x.size(0)
        num_nodes = x.size(1)
        
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.dim_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.dim_head).transpose(1, 2)
        
        distance_mask = self.distance_masking(node_coords)  # [num_nodes, num_nodes, num_heads]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5)  # [batch_size, num_heads, num_nodes, num_nodes]
        
        distance_mask = distance_mask.permute(2, 0, 1).unsqueeze(0)
        
        distance_mask = distance_mask.expand(batch_size, -1, -1, -1)
        
        scores = scores + distance_mask
        
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.num_heads * self.dim_head)
        out = self.out_proj(out)
        
        return out


class TwinningEdgeAttention(nn.Module):
    def __init__(self, dim_edge, dim_node):
        super().__init__()
        self.edge_to_node = build_mlp([dim_edge*2, dim_edge, dim_node], do_bn=False, on_last=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, edge_features, edge_index, num_nodes):
        
        subject_edges = {}
        object_edges = {}
        
        for i, (src, dst) in enumerate(edge_index.t()):
            src, dst = src.item(), dst.item()
            
            if src not in subject_edges:
                subject_edges[src] = []
            subject_edges[src].append(edge_features[i])
            
            if dst not in object_edges:
                object_edges[dst] = []
            object_edges[dst].append(edge_features[i])
        
        node_edge_features = []
        for i in range(num_nodes):
            if i in subject_edges:
                subject_feat = torch.stack(subject_edges[i]).mean(dim=0)
            else:
                subject_feat = torch.zeros_like(edge_features[0])
            
            if i in object_edges:
                object_feat = torch.stack(object_edges[i]).mean(dim=0)
            else:
                object_feat = torch.zeros_like(edge_features[0])
            
            combined = torch.cat([subject_feat, object_feat], dim=0)
            node_edge_features.append(combined)
        
        node_edge_features = torch.stack(node_edge_features)
        
        attention = self.sigmoid(self.edge_to_node(node_edge_features))
        
        return attention


class NodeUpdate(nn.Module):
    def __init__(self, dim_node, dim_edge, num_heads, dim_head):
        super().__init__()
        self.mhsa = DistanceAwareMultiHeadAttention(dim_node, num_heads, dim_head)
        self.twinning_edge_attention = TwinningEdgeAttention(dim_edge, dim_node)
        self.activation = nn.ReLU()
    
    def forward(self, node_features, edge_features, edge_index, node_coords):

        node_features_updated = self.mhsa(node_features, node_coords)
        
        edge_attention = self.twinning_edge_attention(edge_features, edge_index, node_features.size(0))
        
        node_features_final = self.activation(node_features_updated) * edge_attention
        
        return node_features_final


class GraphEdgeAttenNetwork(torch.nn.Module):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, aggr='max', use_bn=False,
                 flow='target_to_source', attention='fat', use_edge:bool=True, **kwargs):
        super().__init__() #  "Max" aggregation.
        self.name = 'edgeatten'
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.index_get = Gen_Index(flow=flow)
        if attention == 'fat':        
            self.index_aggr = Aggre_Index(aggr=aggr, flow=flow)
        elif attention == 'distance':
            aggr = 'add'
            self.index_aggr = Aggre_Index(aggr=aggr, flow=flow)
        else:
            raise NotImplementedError()

        self.edgeatten = MultiHeadedEdgeAttention(
            dim_node=dim_node, dim_edge=dim_edge, dim_atten=dim_atten,
            num_heads=num_heads, use_bn=use_bn, attention=attention, use_edge=use_edge, **kwargs)
        self.prop = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                            do_bn=use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index, weight=None, istrain=False):
        assert x.ndim == 2
        assert edge_feature.ndim == 2
        x_i, x_j = self.index_get(x, edge_index)
        
        edge_dict = {}
        for idx, (src, dst) in enumerate(edge_index.t()):
            edge_dict[(src.item(), dst.item())] = idx
            
        reverse_edge_feature = torch.zeros_like(edge_feature)
        for idx, (src, dst) in enumerate(edge_index.t()):
            if (dst.item(), src.item()) in edge_dict:
                reverse_idx = edge_dict[(dst.item(), src.item())]
                reverse_edge_feature[idx] = edge_feature[reverse_idx]
        
        xx, gcn_edge_feature, prob = self.edgeatten(x_i, edge_feature, x_j, reverse_edge_feature, weight, istrain=istrain)
        xx = self.index_aggr(xx, edge_index, dim_size=x.shape[0])
        xx = self.prop(torch.cat([x, xx], dim=1))
        return xx, gcn_edge_feature


class MultiHeadedEdgeAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_node: int, dim_edge: int, dim_atten: int, use_bn=False,
                 attention='fat', use_edge:bool=True, **kwargs):
        super().__init__()
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.name = 'MultiHeadedEdgeAttention'
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.d_n = d_n = dim_node // num_heads
        self.d_e = d_e = dim_edge // num_heads
        self.d_o = d_o = dim_atten // num_heads
        self.num_heads = num_heads
        self.use_edge = use_edge
        
        self.nn_edge = build_mlp([dim_node*2+dim_edge*2, (dim_node+dim_edge), dim_edge],
                          do_bn=use_bn, on_last=False)
        self.mask_obj = 0.5
        
        DROP_OUT_ATTEN = None
        if 'DROP_OUT_ATTEN' in kwargs:
            DROP_OUT_ATTEN = kwargs['DROP_OUT_ATTEN']
        
        self.attention = attention
        assert self.attention in ['fat']
        
        if self.attention == 'fat':
            if use_edge:
                self.nn = MLP([d_n+d_e, d_n+d_e, d_o], do_bn=use_bn, drop_out=DROP_OUT_ATTEN)
            else:
                self.nn = MLP([d_n, d_n*2, d_o], do_bn=use_bn, drop_out=DROP_OUT_ATTEN)
                
            self.proj_edge = build_mlp([dim_edge, dim_edge])
            self.proj_query = build_mlp([dim_node, dim_node])
            self.proj_value = build_mlp([dim_node, dim_atten])
        elif self.attention == 'distance':
            self.proj_value = build_mlp([dim_node, dim_atten])
    
    def forward(self, query, edge, value, reverse_edge=None, weight=None, istrain=False):
        batch_dim = query.size(0)
        
        if reverse_edge is not None:
            edge_feature = torch.cat([query, edge, reverse_edge, value], dim=1)
        else:
            edge_feature = torch.cat([query, edge, value], dim=1)
        
        edge_feature = self.nn_edge(edge_feature)

        if self.attention == 'fat':
            value = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads)
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)
            if self.use_edge:
                prob = self.nn(torch.cat([query, edge], dim=1)) # b, dim, head    
            else:
                prob = self.nn(query) # b, dim, head 
            prob = prob.softmax(1)
            x = torch.einsum('bm,bm->bm', prob.reshape_as(value), value)
        
        elif self.attention == 'distance':
            raise NotImplementedError()
        
        else:
            raise NotImplementedError('')
        
        return x, edge_feature, prob


class MMG_single(torch.nn.Module):
    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr='max', 
                 use_bn=False, flow='target_to_source', attention='fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, **kwargs):
        
        super().__init__()
        
        self.num_heads = num_heads
        self.depth = depth
        
        self.node_updates = torch.nn.ModuleList()
        self.edge_updates = torch.nn.ModuleList()
        
        for _ in range(self.depth):
            self.node_updates.append(NodeUpdate(
                dim_node=dim_node,
                dim_edge=dim_edge,
                num_heads=num_heads,
                dim_head=dim_atten // num_heads
            ))
            
            self.edge_updates.append(GraphEdgeAttenNetwork(
                num_heads,
                dim_node,
                dim_edge,
                dim_atten,
                aggr,
                use_bn=use_bn,
                flow=flow,
                attention=attention,
                use_edge=use_edge, 
                **kwargs
            ))
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
    
    def forward(self, obj_feature_3d, edge_feature_3d, edge_index, batch_ids, obj_center=None, istrain=False):
        
        for i in range(self.depth):
            _, edge_feature_updated = self.edge_updates[i](
                obj_feature_3d, edge_feature_3d, edge_index, istrain=istrain
            )
            
            if obj_center is not None:
                obj_feature_updated = self.node_updates[i](
                    obj_feature_3d, edge_feature_3d, edge_index, obj_center
                )
            else:
                obj_feature_updated, _ = self.edge_updates[i](
                    obj_feature_3d, edge_feature_3d, edge_index, istrain=istrain
                )
            
            obj_feature_3d = obj_feature_updated
            edge_feature_3d = edge_feature_updated
            
            if i < (self.depth-1) or self.depth==1:
                obj_feature_3d = F.relu(obj_feature_3d)
                obj_feature_3d = self.drop_out(obj_feature_3d)
                
                edge_feature_3d = F.relu(edge_feature_3d)
                edge_feature_3d = self.drop_out(edge_feature_3d)
        
        return obj_feature_3d, edge_feature_3d