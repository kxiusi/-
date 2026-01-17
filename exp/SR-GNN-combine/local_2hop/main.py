#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *
import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import pickle
import time

from entmax import sparsemax, entmax15, entmax_bisect

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mooc_cube', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=5, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')

opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'mooc':
        n_node = 912
    elif opt.dataset == 'mooc_cube':
        n_node = 681
    elif opt.dataset == 'mooc_cube_video_10':
        n_node = 17466
    elif opt.dataset == 'mooc_cube_video_15':
        n_node = 20307
    else:
        n_node = 310

    model = trans_to_cuda(CombineGraph(opt, n_node))
    start = time.time()
    best_result_10 = [0, 0]
    best_epoch_10 = [0, 0]
    bad_counter_10 = 0
    best_result_20 = [0, 0]
    best_epoch_20 = [0, 0]
    bad_counter_20 = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit10, mrr10, hit20, mrr20 = train_test(model, train_data, test_data)
        flag = 0
        if hit10 >= best_result_10[0]:
            best_result_10[0] = hit10
            best_epoch_10[0] = epoch
            flag = 1
        if mrr10 >= best_result_10[1]:
            best_result_10[1] = mrr10
            best_epoch_10[1] = epoch
            flag = 1
        if hit20 >= best_result_20[0]:
            best_result_20[0] = hit20
            best_epoch_20[0] = epoch
            flag = 1
        if mrr20 >= best_result_20[1]:
            best_result_20[1] = mrr20
            best_epoch_20[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tEpoch:\t%d,\t%d\n' % (
            best_result_10[0], best_result_10[1], best_epoch_10[0], best_epoch_10[1]))
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result_20[0], best_result_20[1], best_epoch_20[0], best_epoch_20[1]))
        bad_counter_20 += 1 - flag
        bad_counter_10 += 1 - flag
        if bad_counter_20 >= opt.patience and bad_counter_10 >= opt.patience:
            break
    print('------------------save model--------------------------')
    # input = torch.ones(1, 3, 224, 224)
    # traced_module = torch.jit.trace(model, input)
    # traced_module.save("din4rec.pt")
    scripted_module = torch.jit.script(model)
    torch.jit.save(scripted_module, 'din4rec.pt')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
