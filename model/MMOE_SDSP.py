import numpy as np
import torch
import heapq
import torch.nn as nn
from SDSP.basic.layers import MLP, EmbeddingLayer
import numpy as np
import random


class MMOE_SDSP(nn.Module):
    """Multi-gate Mixture-of-Experts model.

    Args:
        features (list): the list of `Feature Class`, training by the expert and tower module.
        domain_num (int): number of domains.
        n_expert (int): the number of expert nets.
        expert_params (dict): the params of all the expert modules, keys include:`{"dims":list, "activation": str, "dropout":float}.
        tower_params_list (list): the list of tower params dict, the keys same as expert_params.
    """

    def __init__(self, features, scaled_batch, domain_num, n_expert, expert_params, tower_params):
        super().__init__()
        self.features = features
        self.domain_num = domain_num
        self.batch_size = sum(scaled_batch)
        self.n_expert = n_expert
        self.embedding = EmbeddingLayer(features)
        self.input_dims = sum([fea.embed_dim for fea in features])
        self.experts = nn.ModuleList(
            nn.ModuleList(
                MLP(self.input_dims, output_layer=False, **expert_params) for _ in range(n_expert[i]))
            for i in range(domain_num)
        )
        self.gates = nn.ModuleList(
            MLP(self.input_dims, output_layer=False, **{
                "dims": [sum(n_expert)],
                "activation": "softmax"
            }) for i in range(domain_num)
        )
        self.towers = nn.ModuleList(MLP(expert_params["dims"][-1], **tower_params) for i in range(self.domain_num))

        self.dis_fn = nn.PairwiseDistance(p=2)
        self.sim_domain = []
        for d1 in range(domain_num):
            self.sim_domain.append([])
            for d2 in range(domain_num):
                self.sim_domain[d1].append(d2)


        self.expert_ranges = []
        start = 0
        for num in self.n_expert:
            self.expert_ranges.append(list(range(start, start + num)))
            start += num

        self.sim_domain_best = None

        self.proto_num = int(10) 
        self.proto_emb = [None for _ in range(self.domain_num)]

        self.proto_encoders = nn.ModuleList([MLP(scaled_batch[d], output_layer=False, **{
            "dims": [self.proto_num]}) for d in range(self.domain_num)])
        self.proto_decoders = nn.ModuleList([MLP(self.proto_num, output_layer=False, **{
            "dims": [scaled_batch[d]]}) for d in range(self.domain_num)])


    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()
        embed_x = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size, input_dims]
        

        expert_outs = []
        gate_outs = []

        for d in range(self.domain_num):
            for expert in self.experts[d]:
                expert_outs.append(expert(embed_x).unsqueeze(1)) 


        expert_outs = torch.cat(expert_outs, dim=1)
        for d in range(self.domain_num):
            gate_outs.append(self.gates[d](embed_x).unsqueeze(-1)) 
            mask_index = [dom for dom in range(self.domain_num) if dom not in self.sim_domain[d]]
            if mask_index:
                mask = torch.zeros_like(gate_outs[0])  
                for mask_dom in mask_index:
                    mask_temp = self.expert_ranges[mask_dom]
                    mask[:, mask_temp] = -float('inf')  
                gate_outs[d] = gate_outs[d] + mask  
                gate_outs[d] = nn.functional.softmax(gate_outs[d], dim=1)  


        ori_sample_emb = [None for _ in range(self.domain_num)]
        new_sample_emb = [_ for _ in range(self.domain_num)]

        mask = []
        final_outs = []
        for d in range(self.domain_num):
            weight_outs = torch.mul(gate_outs[d], expert_outs)
            emb_outs = torch.sum(weight_outs, dim=1)
            domain_mask = (domain_id == d)
            ori_sample_emb[d] = emb_outs[domain_mask].clone().detach()
            mask.append(domain_mask)
            tower_out = self.towers[d](emb_outs)
            y = torch.sigmoid(tower_out)
            final_outs.append(y)

        
        final = torch.zeros_like(final_outs[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), final_outs[d], final)
        
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


