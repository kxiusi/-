#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mooc', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()

print("-- Starting @ %ss" % datetime.datetime.now())


def obtain_prerequisites():
    item_dict = {}
    with open('mooc_cube_dic.csv', 'r')as f:
        reader = csv.DictReader(f, fieldnames=['key', 'val'], delimiter=',')
        for data in reader:
            key, val = data['key'], data['val']
            item_dict[key] = val

    pre_seq = []
    # dataset = 'course_dense_12.link'
    dataset = 'course.link'
    with open(dataset, "r") as f:
        reader = csv.DictReader(f, fieldnames=['pre', 'suc'], delimiter=' ')

        for data in reader:
            if data['pre'] in item_dict.keys():
                pre = item_dict[data['pre']]
                if data['suc'] in item_dict.keys():
                    suc = item_dict[data['suc']]
                    pre_seq.append([pre, suc])
    print(pre_seq)
    return pre_seq


pre_seqs = obtain_prerequisites()

all = 0

for seq in pre_seqs:
    all += len(seq)

print('avg length: ', all / (len(pre_seqs)))
if not os.path.exists('mooc_cube'):
    os.makedirs('mooc_cube')
# pickle.dump(tra, open('mooc_cube/train.txt', 'wb'))
# pickle.dump(tes, open('mooc_cube/test.txt', 'wb'))
# pickle.dump(tra_seqs, open('mooc_cube/all_train_seq.txt', 'wb'))
# pickle.dump(pre_seqs,open('mooc_cube/prerequisites_dense_12.txt', 'wb'))
pickle.dump(pre_seqs, open('mooc_cube/prerequisites.txt', 'wb'))
print('Done.')
