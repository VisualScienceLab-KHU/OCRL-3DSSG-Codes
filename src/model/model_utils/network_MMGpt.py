import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model_utils.network_util import (MLP, Aggre_Index, Gen_Index,
                                                build_mlp)
from src.model.transformer.attention import MultiHeadAttention


class GraphEdgeAttenNetwork(torch.nn.Module):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, aggr='max', use_bn=False,
                 flow='target_to_source', attention='fat', use_edge:bool=True, **kwargs):
        super().__init__()
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

        self.edge_gate = nn.Sequential(
            nn.Linear(dim_edge, dim_edge // 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim_edge // 2),
            nn.Linear(dim_edge // 2, 1),
            nn.Sigmoid()
        )
        
        self.edgeatten = MultiHeadedEdgeAttention(
            dim_node=dim_node, dim_edge=dim_edge, dim_atten=dim_atten,
            num_heads=num_heads, use_bn=use_bn, attention=attention, use_edge=use_edge, **kwargs)
        
        self.prop = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                            do_bn=use_bn, on_last=False)
        self.layer_norm = nn.LayerNorm(dim_node)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_feature, edge_index, weight=None, istrain=False):
        assert x.ndim == 2
        assert edge_feature.ndim == 2
        x_i, x_j = self.index_get(x, edge_index)
        
        edge_dict = {}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_dict[(src, dst)] = i
        
        reverse_edge_feature = torch.zeros_like(edge_feature)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if (dst, src) in edge_dict:
                reverse_idx = edge_dict[(dst, src)]
                reverse_edge_feature[i] = edge_feature[reverse_idx]
        
        gates = self.edge_gate(edge_feature)
        reverse_edge_feature = gates * reverse_edge_feature
        
        xx, gcn_edge_feature, prob = self.edgeatten(x_i, edge_feature, reverse_edge_feature, x_j, weight, istrain=istrain)
        
        subject_edges = {}
        object_edges = {}
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            
            if src not in subject_edges:
                subject_edges[src] = []
            subject_edges[src].append(i)
            
            if dst not in object_edges:
                object_edges[dst] = []
            object_edges[dst].append(i)
        
        xx = self.index_aggr(xx, edge_index, dim_size=x.shape[0])
        
        twinning_edge_attention = torch.zeros_like(xx)
        for node_idx in range(x.shape[0]):
            subj_features = []
            if node_idx in subject_edges:
                for edge_idx in subject_edges[node_idx]:
                    subj_features.append(gcn_edge_feature[edge_idx])
            
            obj_features = []
            if node_idx in object_edges:
                for edge_idx in object_edges[node_idx]:
                    obj_features.append(gcn_edge_feature[edge_idx])
            
            if subj_features:
                subj_agg = torch.stack(subj_features).mean(dim=0)
            else:
                subj_agg = torch.zeros(self.dim_edge, device=x.device)
                
            if obj_features:
                obj_agg = torch.stack(obj_features).mean(dim=0)
            else:
                obj_agg = torch.zeros(self.dim_edge, device=x.device)
            
            edge_agg = torch.cat([subj_agg, obj_agg])
            
            twinning_edge_attention[node_idx] = nn.Linear(edge_agg.shape[0], xx.shape[1], device=x.device)(edge_agg)
        
        xx = F.relu(xx) * self.sigmoid(twinning_edge_attention)
        
        xx = self.prop(torch.cat([x, xx], dim=1))
        xx = self.layer_norm(xx)
        
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
        
        self.nn_edge = build_mlp([dim_node*2+dim_edge*2, (dim_node+dim_edge*2), dim_edge],
                          do_bn=use_bn, on_last=False)
        self.edge_layer_norm = nn.LayerNorm(dim_edge)
        self.mask_obj = 0.5
        
        DROP_OUT_ATTEN = kwargs.get('DROP_OUT_ATTEN', 0.5)  
        
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

        
    def forward(self, query, edge, reverse_edge, value, weight=None, istrain=False):
        batch_dim = query.size(0)
        
        edge_feature = torch.cat([query, edge, reverse_edge, value], dim=1)
        edge_feature = self.nn_edge(edge_feature)
        edge_feature = self.edge_layer_norm(edge_feature)

        if self.attention == 'fat':
            value = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads)
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)
            if self.use_edge:
                prob = self.nn(torch.cat([query, edge], dim=1))  # b, dim, head    
            else:
                prob = self.nn(query)  # b, dim, head
                
            prob = prob.softmax(1)
            x = torch.einsum('bm,bm->bm', prob.reshape_as(value), value)
        
        elif self.attention == 'distance':
            raise NotImplementedError()
        
        else:
            raise NotImplementedError('')
        
        return x, edge_feature, prob
    

class MMG_pt_single(torch.nn.Module):
    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr='max', 
                 use_bn=False, flow='target_to_source', attention='fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, **kwargs):
        
        super().__init__()

        self.num_heads = num_heads
        self.depth = depth

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads) 
            for i in range(depth)
        )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim_node) for _ in range(depth)
        ])

        self.gcn_3ds = torch.nn.ModuleList()
        
        for _ in range(self.depth):
            self.gcn_3ds.append(GraphEdgeAttenNetwork(
                            num_heads,
                            dim_node,
                            dim_edge,
                            dim_atten,
                            aggr,
                            use_bn=use_bn,
                            flow=flow,
                            attention=attention,
                            use_edge=use_edge, 
                            **kwargs))
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
        self.self_attn_fc = nn.Sequential(  # 11 32 32 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads)
        )
        
        self.residual_proj = nn.ModuleList([
            nn.Linear(dim_node, dim_node) for _ in range(depth)
        ])
    
    def forward(self, obj_feature_3d, edge_feature_3d, edge_index, batch_ids, obj_center=None, istrain=False):
        
        if obj_center is not None:
            batch_size = batch_ids.max().item() + 1
            N_K = obj_feature_3d.shape[0]
            obj_mask = torch.zeros(1, 1, N_K, N_K).cuda()
            obj_distance_weight = torch.zeros(1, self.num_heads, N_K, N_K).cuda()
            count = 0

            for i in range(batch_size):
                idx_i = torch.where(batch_ids == i)[0]
                obj_mask[:, :, count:count + len(idx_i), count:count + len(idx_i)] = 1
            
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(len(idx_i), 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, len(idx_i), 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)  # 1 N N 4
                dist_weights = self.self_attn_fc(weights).permute(0, 3, 1, 2)  # 1 num_heads N N
                
                attention_matrix_way = 'add'
                obj_distance_weight[:, :, count:count + len(idx_i), count:count + len(idx_i)] = dist_weights

                count += len(idx_i)
        else:
            obj_mask = None
            obj_distance_weight = None
            attention_matrix_way = 'mul'
        
        for i in range(self.depth):
            identity_obj = self.residual_proj[i](obj_feature_3d)
            
            obj_feature_3d = obj_feature_3d.unsqueeze(0)
            obj_feature_attn = self.self_attn[i](
                obj_feature_3d, obj_feature_3d, obj_feature_3d, 
                attention_weights=obj_distance_weight, way=attention_matrix_way, 
                attention_mask=obj_mask, use_knn=False
            )
            obj_feature_attn = obj_feature_attn.squeeze(0)
            
            obj_feature_attn = self.layer_norms[i](obj_feature_attn)
            
            obj_feature_3d_new, edge_feature_3d_new = self.gcn_3ds[i](obj_feature_attn, edge_feature_3d, edge_index, istrain=istrain)
            
            obj_feature_3d = obj_feature_3d_new + identity_obj
            edge_feature_3d = edge_feature_3d_new
            
            if i < (self.depth-1) or self.depth==1:
                obj_feature_3d = F.relu(obj_feature_3d)
                obj_feature_3d = self.drop_out(obj_feature_3d)
                
                edge_feature_3d = F.relu(edge_feature_3d)
                edge_feature_3d = self.drop_out(edge_feature_3d)
        
        return obj_feature_3d, edge_feature_3d