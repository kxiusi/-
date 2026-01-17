import pickle
import numpy as np
import pandas as pd

PATH_TO_TRAIN = 'processed_data/mooc_cube_train_full_new.txt'
train_data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
f = open('mooc/item_knn_new.pkl', 'rb')
item_knn = pickle.load(f)
items_to_predict = train_data['ItemId'].unique()

test_data = [213, 25, 12, 29, 5, 3, 2, 57, 211, 28, 113, 18, 141, 38]
prev_iid = test_data[-1]
preds = item_knn.predict_next(None,prev_iid, items_to_predict)
preds[np.isnan(preds)] = 0
sorted_preds = preds.nlargest(20)
print(sorted_preds.index.tolist())
