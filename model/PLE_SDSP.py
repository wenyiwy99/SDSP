"""
References: 
paper: (RecSys'2020) Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
url: https://dl.acm.org/doi/abs/10.1145/3383313.3412236
"""

import torch
import torch.nn as nn

from SDSP.basic.layers import MLP, EmbeddingLayer
import heapq

class PLE_SDSP(nn.Module):
    """Progressive-Layered Extraction model.

    Args:
        features (list): the list of `Feature Class`, training by the expert and tower module.
        domain_num (int): domain numbers.
        n_level (int): the number of CGC layers.
        n_expert_specific (int): the number of task-specific expert nets.
        n_expert_shared (int): the number of task-shared expert nets.
        expert_params (dict): the params of all the expert modules, keys include:`{"dims":list, "activation": str, "dropout": float}.
        tower_params (dict): the list of tower params dict, the keys same as expert_params.
    """

    def __init__(self, features, scaled_batch, domain_num, n_level, n_expert_specific, n_expert_shared, expert_params,
                 tower_params):
        super().__init__()
        self.features = features
        self.domain_num = domain_num
        self.scaled_batch = scaled_batch
        self.n_level = n_level
        self.input_dims = sum([fea.embed_dim for fea in features])
        self.embedding = EmbeddingLayer(features)
        self.cgc_layers = nn.ModuleList(
            CGC(i + 1, n_level, self.domain_num, n_expert_specific, n_expert_shared, self.input_dims, expert_params)
            for i in range(n_level))
        self.towers = nn.ModuleList(
            MLP(expert_params["dims"][-1], output_layer=True, **tower_params) for i in range(self.domain_num))
        
        self.dis_fn = nn.PairwiseDistance(p=2)
        self.sim_domain = []
        for d1 in range(domain_num):
            self.sim_domain.append([])
            for d2 in range(domain_num):
                self.sim_domain[d1].append(d2)
        self.sim_domain_best = [[] for _ in range(domain_num)]
        

        self.proto_num = int(10)
        self.proto_emb = [None for _ in range(self.domain_num)]
        self.proto_encoders = nn.ModuleList([MLP(scaled_batch[d], output_layer=False, **{
            "dims": [self.proto_num]}) for d in range(self.domain_num)])
        self.proto_decoders = nn.ModuleList([MLP(self.proto_num, output_layer=False, **{
            "dims": [scaled_batch[d]]}) for d in range(self.domain_num)])

        # self.predict_layers = nn.ModuleList(PredictionLayer(task_type) for task_type in task_types)

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()
        embed_x = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size, input_dims]
    
        ple_inputs = [embed_x] * (self.domain_num + 1)
        ple_outs = []
        for i in range(self.n_level):
            ple_outs = self.cgc_layers[i](ple_inputs, self.sim_domain)  # ple_outs[i]: [batch_size, expert_dims[-1]]
            ple_inputs = ple_outs


        ori_sample_emb = [None for _ in range(self.domain_num)]
        new_sample_emb = [None for _ in range(self.domain_num)]
            
        ys = []
        mask = []
        d = 0

        for ple_out, tower in zip(ple_outs, self.towers):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)
            tower_out = tower(ple_out)  # [batch_size, 1]
            y = torch.sigmoid(tower_out)  # logit -> proba
            ori_sample_emb[d] = ple_outs[d][domain_mask].clone().detach()
            d += 1
            ys.append(y)

        final = torch.zeros_like(ys[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), ys[d], final)
        if self.training:
            for d in range(self.domain_num):
                proto_emb = self.proto_encoders[d](torch.transpose(ori_sample_emb[d], 0, 1))
                new_sample_emb[d] = torch.transpose(self.proto_decoders[d](proto_emb), 0, 1)
                self.proto_emb[d] = torch.transpose(proto_emb, 0, 1)  

            return final.squeeze(1), ori_sample_emb, new_sample_emb
        else:

            return final.squeeze(1)

    def domain_distance(self, proto_emb, sim_num, target_dom):
        sim_domain = []
        dom_dis = [None for _ in range(self.domain_num)]
        for d2 in range(self.domain_num):
            if d2 == target_dom:
                dom_dis[d2]= 0
                continue
            final_dis = 0
            for i in range(self.proto_num):
                min_distance = float('inf')
                for j in range(self.proto_num):
                    distance = self.dis_fn(proto_emb[target_dom][i], proto_emb[d2][j])
                    if distance < min_distance:
                        min_distance = distance
                final_dis += min_distance
            final_dis /= self.proto_num
            dom_dis[d2] = final_dis.item()
        temp_list = heapq.nsmallest(sim_num + 1, dom_dis)
        for d2 in range(self.domain_num):
            if dom_dis[d2] in temp_list:
                sim_domain.append(d2)
        return sim_domain

class CGC(nn.Module):
    """Customized Gate Control (CGC) Model mentioned in PLE paper.

    Args:
        cur_level (int): the current level of CGC in PLE.
        n_level (int): the number of CGC layers.
        domain_num (int): the number of domain_num.
        n_expert_specific (int): the number of task-specific expert nets.
        n_expert_shared (int): the number of task-shared expert nets.
        input_dims (int): the input dimes of the expert module in the current CGC layer.
        expert_params (dict): the params of all the expert modules, keys include:`{"dims":list, "activation": str, "dropout":float}.
    """

    def __init__(self, cur_level, n_level, domain_num, n_expert_specific, n_expert_shared_list, input_dims, expert_params):
        super().__init__()
        self.cur_level = cur_level  # the CGC level of PLE
        self.n_level = n_level
        self.domain_num = domain_num
        self.n_expert_specific = n_expert_specific
        self.n_expert_shared = sum(n_expert_shared_list)
        self.n_expert_all = n_expert_specific * self.domain_num + self.n_expert_shared
        input_dims = input_dims if cur_level == 1 else expert_params["dims"][-1]
        self.experts_specific = nn.ModuleList(
            MLP(input_dims, output_layer=False, **expert_params) for _ in range(self.domain_num * self.n_expert_specific))
        self.experts_shared = nn.ModuleList(
            MLP(input_dims, output_layer=False, **expert_params) for _ in range(self.n_expert_shared))
        self.gates_specific = nn.ModuleList(
            MLP(
                input_dims, **{
                    "dims": [self.n_expert_specific + self.n_expert_shared],
                    "activation": "softmax",
                    "output_layer": False
                }) for _ in range(self.domain_num))
        if cur_level < n_level:
            self.gate_shared = MLP(input_dims, **{
                "dims": [self.n_expert_all],
                "activation": "softmax",
                "output_layer": False
            })  # n_gate_specific = n_task

        self.expert_ranges = []
        start = n_expert_specific
        for num in n_expert_shared_list:
            self.expert_ranges.append(list(range(start, start + num)))
            start += num

    def forward(self, x_list, sim_domain):
        expert_specific_outs = []
        for i in range(self.domain_num):
            expert_specific_outs.extend([
                expert(x_list[i]).unsqueeze(1)
                for expert in self.experts_specific[i * self.n_expert_specific:(i + 1) * self.n_expert_specific]
            ])
        expert_shared_outs = [expert(x_list[-1]).unsqueeze(1) for expert in self.experts_shared
                             ]  # x_list[-1]: the input for shared experts
        gate_specific_outs = [gate(x_list[i]).unsqueeze(-1) for i, gate in enumerate(self.gates_specific)
                             ]  # gate_out[i]: [batch_size, n_expert_specific+n_expert_shared, 1]
        mask_index = [None for _ in range(self.domain_num)]

        cgc_outs = []
        for i, gate_out in enumerate(gate_specific_outs):
            cur_expert_list = expert_specific_outs[i * self.n_expert_specific:(i + 1) *
                                                   self.n_expert_specific] + expert_shared_outs
            expert_concat = torch.cat(cur_expert_list,
                                      dim=1)  # [batch_size, n_expert_specific+n_expert_shared, expert_dims[-1]]
            mask_index = [dom for dom in range(self.domain_num) if dom not in sim_domain[i]]
            if mask_index:
                mask = torch.zeros_like(gate_out)  # 初始化掩码
                for mask_dom in mask_index:
                    mask_temp = self.expert_ranges[mask_dom]
                    mask[:, mask_temp] = -float('inf')  
                gate_out = gate_out + mask  # 应用掩码
                gate_out = nn.functional.softmax(gate_out, dim=1)
            expert_weight = torch.mul(gate_out,
                                      expert_concat)  # [batch_size, n_expert_specific+n_expert_shared, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)  # [batch_size, expert_dims[-1]]
            cgc_outs.append(expert_pooling)  # length: n_task
        if self.cur_level < self.n_level:  # not the last layer
            gate_shared_out = self.gate_shared(x_list[-1]).unsqueeze(-1)  # [batch_size, n_expert_all, 1]
            expert_concat = torch.cat(expert_specific_outs + expert_shared_outs,
                                      dim=1)  # [batch_size, n_expert_all, expert_dims[-1]]
            expert_weight = torch.mul(gate_shared_out, expert_concat)  # [batch_size, n_expert_all, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)  # [batch_size, expert_dims[-1]]
            cgc_outs.append(expert_pooling)  # length: n_task+1

        return cgc_outs

