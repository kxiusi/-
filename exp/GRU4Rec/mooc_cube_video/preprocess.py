# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:20:12 2015

@author: Balázs Hidasi
"""

import numpy as np
import pandas as pd
import datetime as dt

PATH_TO_ORIGINAL_DATA = '/home/jywang/code/mooccube_process/'
PATH_TO_PROCESSED_DATA = '/home/jywang/code/GRU4Rec/processed_data/'

data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'mooc_cube_video_10.csv', sep=',', header=None, usecols=[0, 1],
                   dtype={0: np.int32, 1: np.int32})
data.columns = ['SessionId', 'ItemId']

session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]
item_supports = data.groupby('ItemId').size()
data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]
session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= 2].index)]

maxSession = session_lengths.size
splitSession = maxSession // 10 * 9
sessionId_max = data.groupby('SessionId').SessionId.max()
session_train = sessionId_max[sessionId_max < splitSession].index
session_test = sessionId_max[sessionId_max >= splitSession].index
train = data[np.in1d(data.SessionId, session_train)]
test = data[np.in1d(data.SessionId, session_test)]
test = test[np.in1d(test.ItemId, train.ItemId)]
tslength = test.groupby('SessionId').size()
test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                         train.ItemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + 'mooc_cube_video_train_full.txt', sep='\t', index=False)
print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                   test.ItemId.nunique()))
test.to_csv(PATH_TO_PROCESSED_DATA + 'mooc_cube_video_test.txt', sep='\t', index=False)

splitTrSession = maxSession // 10 * 8
sessionId_max = train.groupby('SessionId').SessionId.max()
session_train = sessionId_max[sessionId_max < splitTrSession].index
session_valid = sessionId_max[sessionId_max >= splitTrSession].index
train_tr = train[np.in1d(train.SessionId, session_train)]
valid = train[np.in1d(train.SessionId, session_valid)]
valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
tslength = valid.groupby('SessionId').size()
valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
                                                                    train_tr.ItemId.nunique()))
train_tr.to_csv(PATH_TO_PROCESSED_DATA + 'mooc_cube_video_train_tr.txt', sep='\t', index=False)
print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                         valid.ItemId.nunique()))
valid.to_csv(PATH_TO_PROCESSED_DATA + 'mooc_cube_video_train_valid.txt', sep='\t', index=False)
