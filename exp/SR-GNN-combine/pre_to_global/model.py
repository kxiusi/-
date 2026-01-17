#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Parameter
from utils import GlobalAggregator


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class CombineGraph(Module):
    def __init__(self, opt, n_node, adj1, num1, adj2, num2):
        super(CombineGraph, self).__init__()

        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.softmax = nn.LogSoftmax()
        self.loss_function = nn.NLLLoss()
        # self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

        # Aggregator
        self.sample_num = opt.n_sample_all
        self.global_agg = GlobalAggregator(self.hidden_size, opt.dropout_gcn, act=torch.relu)

        #
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout_local = nn.Dropout(opt.dropout_local)
        self.dropout_global = nn.Dropout(opt.dropout_global)

        self.adj1 = trans_to_cuda(torch.Tensor(adj1)).long()
        self.num1 = trans_to_cuda(torch.Tensor(num1)).float()
        self.adj2 = trans_to_cuda(torch.Tensor(adj2)).long()
        self.num2 = trans_to_cuda(torch.Tensor(num2)).float()

    def sample(self, target, adj, num):
        return adj[target.view(-1)], num[target.view(-1)]

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A1, A2, mask_item, item):

        hidden = self.embedding(inputs)
        # local
        hidden1 = self.gnn(A1, hidden)
        hidden2 = self.gnn(A2, hidden)
        alpha1 = torch.sigmoid(self.linear1(hidden1))
        alpha2 = torch.sigmoid(self.linear1(hidden2))
        h_local = alpha1 * hidden1 + alpha2 * hidden2

        # global
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        item_self = [inputs]
        weight_neighbors = []
        item_neighbors = []

        item_sample_i, weight_sample_i = self.sample(item_self[0], self.adj1, self.num1)
        item_neighbors.append(item_sample_i.view(batch_size, -1, self.sample_num))
        weight_neighbors.append(weight_sample_i.view(batch_size, -1, self.sample_num))

        entity_vectors = [self.embedding(item_self[0])]
        neighbor_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)

        # mean
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)

        # sum
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        session_info = sum_item_emb.repeat(1, entity_vectors[0].shape[1], 1)
        aggregator = self.global_agg
        shape = [batch_size, -1, self.sample_num, self.hidden_size]
        # print(neighbor_vectors[0].shape)
        # print(neighbor_vectors[1].shape)
        # print(session_info.shape)
        h_global = aggregator(self_vectors=entity_vectors[0],
                              neighbor_vector=neighbor_vectors[0].view(shape),
                              masks=None,
                              batch_size=batch_size,
                              neighbor_weight=weight_vectors[0].view(batch_size, -1, self.sample_num),
                              extra_vector=session_info)

        # dep
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        item_self = [inputs]
        weight_neighbors = []
        item_neighbors = []

        item_sample_i, weight_sample_i = self.sample(item_self[0], self.adj2, self.num2)
        item_neighbors.append(item_sample_i.view(batch_size, -1, self.sample_num))
        weight_neighbors.append(weight_sample_i.view(batch_size, -1, self.sample_num))

        entity_vectors = [self.embedding(item_self[0])]
        neighbor_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)

        # mean
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)

        # sum
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        session_info = sum_item_emb.repeat(1, entity_vectors[0].shape[1], 1)
        aggregator = self.global_agg
        shape = [batch_size, -1, self.sample_num, self.hidden_size]
        # print(neighbor_vectors[0].shape)
        # print(neighbor_vectors[1].shape)
        # print(session_info.shape)
        h_dep = aggregator(self_vectors=entity_vectors[0],
                           neighbor_vector=neighbor_vectors[0].view(shape),
                           masks=None,
                           batch_size=batch_size,
                           neighbor_weight=weight_vectors[0].view(batch_size, -1, self.sample_num),
                           extra_vector=session_info)

        # alpha1 = torch.sigmoid(self.linear2(vector1))
        # alpha2 = torch.sigmoid(self.linear2(vector2))
        # h_global = alpha1 * vector1 + alpha2 * vector2
        #   # combine
        # h_local = self.dropout_local(hidden1)
        # h_global = vector1 + vector2
        # h_global = self.dropout_global(h_global)
        # h_global = vector1
        # return h_global
        output = h_local + h_dep
        return output
        # return vector1


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A1, A2, items, mask, targets, inputs = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A1 = np.array(A1)
    A1 = trans_to_cuda(torch.Tensor(A1).float())
    A2 = np.array(A2)
    A2 = trans_to_cuda(torch.Tensor(A2).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    inputs = trans_to_cuda(torch.Tensor(inputs)).long()
    hidden = model(items, A1, A2, mask, inputs)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        # print("training scroes:")
        targets = trans_to_cuda(torch.Tensor(targets).long())

        softmax = model.softmax(scores)
        loss = model.loss_function(softmax, targets - 1)
        # alpha_ent = torch.tensor(1.5, requires_grad=True)
        # sparsemax_output = entmax_bisect(scores)
        # sparsemax_output = sparsemax(scores)
        # loss = model.loss_function(sparsemax_output, targets - 1)

        loss.backward()
        model.optimizer.step()
        total_loss += loss
        # if j % int(len(slices) / 5 + 1) == 0:
        # print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit10, mrr10 = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(10)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit10 = np.mean(hit10) * 100
    mrr10 = np.mean(mrr10) * 100

    hit20, mrr20 = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr20.append(0)
            else:
                mrr20.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit20 = np.mean(hit20) * 100
    mrr20 = np.mean(mrr20) * 100
    return hit10, mrr10, hit20, mrr20
