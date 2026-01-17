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

dataset = 'mooc.csv'
# dataset = 'mooc_cube_video.csv'
print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    reader = csv.DictReader(f, fieldnames=['session_id', 'timestamp', 'item_id'], delimiter=',')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    tag = False
    n_node = []
    for data in reader:
        sessid = data['session_id']
        curid = sessid
        item = data['item_id']
        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
print("-- Reading data @ %ss" % datetime.datetime.now())
# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
    if len(sess_clicks[s]) > 10:
        sess_clicks[s] = sess_clicks[s][-10:]
# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
clicks = list(sess_clicks.items())
print(clicks[:10])

maxsession = length

splitsession = maxsession // 10 * 9
print('Splitting session', splitsession)
# tra_sess = filter(lambda x: int(x[0]) < splitsession, clicks)
# tes_sess = filter(lambda x: int(x[0]) > splitsession, clicks)

# Sort sessions by date
tra_sess = clicks[:splitsession]
# sorted(tra_sess, key=operator.itemgetter(0))     # [(session_id, timestamp), (), ]
tes_sess = clicks[splitsession:]
# sorted(tes_sess, key=operator.itemgetter(0))     # [(session_id, timestamp), (), ]
print(len(tra_sess))  # 186670    # 7966257
print(len(tes_sess))  # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count_course >=5 gives approximately the same number of items as reported in paper
item_dict = {}


# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    item_ctr = 1
    for s, _ in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_seqs += [outseq]
    print(item_ctr)  # 43098, 37484
    return train_ids, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    for s, _ in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_seqs += [outseq]
    return test_ids, test_seqs


tra_ids, tra_seqs = obtian_tra()
tes_ids, tes_seqs = obtian_tes()


# print(tra_ids[:5],tra_seqs[:5])
# print(tes_ids[:5], tes_seqs[:5])
def process_seqs(iseqs):
    out_seqs = []
    labs = []
    ids = []
    for id, seq in zip(range(len(iseqs)), iseqs):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            ids += [id]
    return out_seqs, labs, ids


tr_seqs, tr_labs, tr_ids = process_seqs(tra_seqs)
te_seqs, te_labs, te_ids = process_seqs(tes_seqs)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:5], tr_labs[:5])
print(te_seqs[:5], te_labs[:5])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all / (len(tra_seqs) + len(tes_seqs) * 1.0))
# if not os.path.exists('mooc_cube_video'):
#     os.makedirs('mooc_cube_video')
# pickle.dump(tra, open('mooc_cube_video/train.txt', 'wb'))
# pickle.dump(tes, open('mooc_cube_video/test.txt', 'wb'))
# pickle.dump(tra_seqs, open('mooc_cube_video/all_train_seq.txt', 'wb'))

print('Done.')
