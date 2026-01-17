#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import pickle

from model import *
import datetime
import time
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mooc_cube_new',
                    help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
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
parser.add_argument('--dropout_gcn', type=float, default=0.2)
parser.add_argument('--dropout_local', type=float, default=0.0)
parser.add_argument('--dropout_global', type=float, default=0.0)
parser.add_argument('--n_sample_all', type=int, default=12)

opt = parser.parse_args()
print(opt)


def model_test(model, test_data):
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit20, mrr20 = [], []
    slices = test_data.generate_batch(1)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_20_score = scores.topk(20)[0]
        sub_20_id = scores.topk(20)[1]

        # if(targets[0]==490 and 490 in sub_20_id.tolist()):
        #     print("targets:", targets)
        #     print("sub_20_id:", sub_20_id)
        #     print("sub_20_score:", sub_20_score)

        sub_scores_20 = scores.topk(20)[1]
        sub_scores_20 = trans_to_cpu(sub_scores_20).detach().numpy()
        for score, target, mask in zip(sub_scores_20, targets, test_data.mask):
            hit20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr20.append(0)
            else:
                mrr20.append(1 / (np.where(score == target - 1)[0][0] + 1))

    hit20 = np.mean(hit20) * 100
    mrr20 = np.mean(mrr20) * 100
    return hit20, mrr20


def main():
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    test_data = Data(test_data, shuffle=False)
    model = torch.jit.load('din4rec.pt', map_location='cpu')
    #model = torch.jit.load('din4rec.pt')

    start = time.time()
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit20, mrr20 = model_test(model, test_data=test_data)
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (
            hit20, mrr20))

    print('-------------------------------------------------------')

    end = time.time()
    print("Run time: %f s" % (end - start))

if __name__ == '__main__':
    main()
