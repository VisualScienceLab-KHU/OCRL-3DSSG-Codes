import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model.model_utils.network_util import (MLP, Aggre_Index, Gen_Index, build_mlp)

class DualAttentionEdgeGAT(torch.nn.Module):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, aggr='max', use_bn=False,
                 flow='target_to_source', use_edge=True, **kwargs):
        super().__init__()
        self.name = 'dualedgeatten'
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_atten = dim_atten
        self.index_get = Gen_Index(flow=flow)
        self.index_aggr = Aggre_Index(aggr=aggr, flow=flow)

        self.edgeatten = DualAttentionMechanism(
            dim_node=dim_node, dim_edge=dim_edge, dim_atten=dim_atten,
            num_heads=num_heads, use_bn=use_bn, use_edge=use_edge, **kwargs)
        
        self.prop = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                            do_bn=use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index, geo_features=None, weight=None, istrain=False):
        assert x.ndim == 2
        assert edge_feature.ndim == 2
        
        print(f"x shape: {x.shape}")
        print(f"edge_feature shape: {edge_feature.shape}")
        
        x_i, x_j = self.index_get(x, edge_index)
        
        xx, gcn_edge_feature, balance = self.edgeatten(
            x_i, edge_feature, x_j, geo_features, weight, istrain=istrain)
        
        print(f"xx shape before aggregation: {xx.shape}")
        
        xx = self.index_aggr(xx, edge_index, dim_size=x.shape[0])
        
        print(f"xx shape after aggregation: {xx.shape}")
        print(f"torch.cat([x, xx], dim=1) shape: {torch.cat([x, xx], dim=1).shape}")
        
        xx = self.prop(torch.cat([x, xx], dim=1))
        
        return xx, gcn_edge_feature, balance


class DualAttentionMechanism(torch.nn.Module):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, use_bn=False, 
                 use_edge=True, **kwargs):
        super().__init__()
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        
        self.name = 'DualAttentionMechanism'
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_atten = dim_atten
        self.d_n = d_n = dim_node // num_heads
        self.d_e = d_e = dim_edge // num_heads
        self.d_o = d_o = dim_atten // num_heads
        self.num_heads = num_heads
        self.use_edge = use_edge
        
        self.nn_edge = build_mlp([dim_node * 2 + dim_edge, (dim_node + dim_edge), dim_edge],
                                  do_bn=use_bn, on_last=False)
        
        DROP_OUT_ATTEN = kwargs.get('DROP_OUT_ATTEN', 0.1)
        
        if use_edge:
            geo_input_dim = d_n + d_e + d_o  # query_proj + edge_proj + geo_proj
            sem_input_dim = d_n + d_e        # query_proj + edge_proj
        else:
            geo_input_dim = d_n + d_o        # query_proj + geo_proj
            sem_input_dim = d_n              # query_proj
        
        print(f"geo_input_dim: {geo_input_dim}, sem_input_dim: {sem_input_dim}")
        print(f"d_n: {d_n}, d_e: {d_e}, d_o: {d_o}, num_heads: {num_heads}")
        
        self.geo_nn = MLP([geo_input_dim, geo_input_dim, d_o], do_bn=use_bn, drop_out=DROP_OUT_ATTEN)
        self.sem_nn = MLP([sem_input_dim, sem_input_dim, d_o], do_bn=use_bn, drop_out=DROP_OUT_ATTEN)
        
        self.balance_network = nn.Sequential(
            nn.Linear(dim_edge, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROP_OUT_ATTEN),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(DROP_OUT_ATTEN),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.proj_edge = build_mlp([dim_edge, dim_edge])
        self.proj_query = build_mlp([dim_node, dim_node])
        self.proj_value = build_mlp([dim_node, dim_atten])
        
        self.proj_geo = build_mlp([11, dim_atten])
        
    def forward(self, query, edge, value, geo_features=None, weight=None, istrain=False):
        batch_dim = query.size(0)
        
        edge_feature = torch.cat([query, edge, value], dim=1)
        edge_feature = self.nn_edge(edge_feature)
        
        value_proj = self.proj_value(value)
        query_proj = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads)
        edge_proj = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)
        
        if geo_features is not None:
            geo_proj = self.proj_geo(geo_features).view(batch_dim, self.d_o, self.num_heads)
            if self.use_edge:
                geo_input = torch.cat([query_proj, edge_proj, geo_proj], dim=1)
            else:
                geo_input = torch.cat([query_proj, geo_proj], dim=1)
            geo_prob = self.geo_nn(geo_input)
        else:
            geo_prob = torch.zeros_like(query_proj[:, :self.d_o, :])
        
        if self.use_edge:
            sem_input = torch.cat([query_proj, edge_proj], dim=1)
        else:
            sem_input = query_proj
        sem_prob = self.sem_nn(sem_input)
        
        geo_prob = F.softmax(geo_prob, dim=1)
        sem_prob = F.softmax(sem_prob, dim=1)
        
        balance = self.balance_network(edge).view(batch_dim, 1, 1)
        
        value_proj_reshaped = value_proj.view(batch_dim, self.d_o, self.num_heads)
        
        print(f"geo_prob shape: {geo_prob.shape}")
        print(f"sem_prob shape: {sem_prob.shape}")
        print(f"value_proj_reshaped shape: {value_proj_reshaped.shape}")
        
        geo_output = torch.sum(geo_prob * value_proj_reshaped, dim=(1, 2))
        sem_output = torch.sum(sem_prob * value_proj_reshaped, dim=(1, 2))
        
        geo_output = geo_output.unsqueeze(1).expand(-1, self.dim_atten)
        sem_output = sem_output.unsqueeze(1).expand(-1, self.dim_atten)
        
        output = balance.squeeze(-1).squeeze(-1).unsqueeze(1) * geo_output + (1 - balance.squeeze(-1).squeeze(-1).unsqueeze(1)) * sem_output
        
        return output, edge_feature, balance.squeeze(-1).squeeze(-1)


class MMG_dual(torch.nn.Module):
    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr='max', 
                 use_bn=False, flow='target_to_source', use_edge=True, 
                 hidden_size=512, depth=1, **kwargs):
        
        super().__init__()
        self.num_heads = num_heads
        self.depth = depth
        
        self.gcn_3ds = torch.nn.ModuleList()
        
        for _ in range(self.depth):
            self.gcn_3ds.append(DualAttentionEdgeGAT(
                num_heads,
                dim_node,
                dim_edge,
                dim_atten,
                aggr,
                use_bn=use_bn,
                flow=flow,
                use_edge=use_edge, 
                **kwargs))
        
        self.drop_out = torch.nn.Dropout(kwargs.get('DROP_OUT_ATTEN', 0.1))
    
    def forward(self, obj_feature_3d, edge_feature_3d, edge_index, batch_ids, 
                obj_center=None, geo_features=None, istrain=False):
        
        balance_values = []
        
        for i in range(self.depth):
            obj_feature_3d, edge_feature_3d, balance = self.gcn_3ds[i](
                obj_feature_3d, edge_feature_3d, edge_index, 
                geo_features=geo_features, istrain=istrain)
            
            balance_values.append(balance)
            
            if i < (self.depth-1) or self.depth==1:
                obj_feature_3d = F.relu(obj_feature_3d)
                obj_feature_3d = self.drop_out(obj_feature_3d)
                
                edge_feature_3d = F.relu(edge_feature_3d)
                edge_feature_3d = self.drop_out(edge_feature_3d)
        
        return obj_feature_3d, edge_feature_3d, balance_values