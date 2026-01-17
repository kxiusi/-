from flask import Flask, request

app = Flask(__name__)
from model import *
from utils import *
import pickle
import numpy as np
import pandas as pd


def predict(seq):
    test_data = [seq]
    test_data = Data(test_data, shuffle=False)
    pretrained_model.eval()
    slices = test_data.generate_batch(1)
    for i in slices:
        scores = forward(pretrained_model, i, test_data)
        sub_20_score = scores.topk(20)[0]
        sub_20_id = scores.topk(20)[1]
        sub_20_id_list = sub_20_id.tolist()[0]
        res = [i + 1 for i in sub_20_id_list]
        # print("ids_20:", res)
        print("scores_20:", sub_20_score)
    return {"results": res}


def itemknn_predict(seq):
    prev_iid = seq[-1]
    preds = item_knn.predict_next(None, prev_iid, items_to_predict)
    preds[np.isnan(preds)] = 0
    sorted_preds = preds.nlargest(20)
    print("scores_20:", sorted_preds)
    return {"results": sorted_preds.index.tolist()}


@app.route('/recommend', methods=["GET", "POST"])
def din4rec():  # put application's code here
    if request.method == 'GET':
        params = request.args
    else:
        params = request.form if request.form else request.json
    seq = params.get("seq", 0)
    print(seq)
    res = predict(seq)
    print(res)
    return res


@app.route('/baselineRecommend', methods=["GET", "POST"])
def baseline():  # put application's code here
    if request.method == 'GET':
        params = request.args
    else:
        params = request.form if request.form else request.json
    seq = params.get("seq", 0)
    print(seq)
    res = itemknn_predict(seq)
    print(res)
    return res


# din4rec
pretrained_model = torch.jit.load('din4rec.pt', map_location='cpu')
# itemknn
PATH_TO_TRAIN = '/Users/dmrfcoder/Documents/毕设/code/实验/GRU4Rec/processed_data/mooc_cube_train_full_new.txt'
train_data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
f = open('/Users/dmrfcoder/Documents/毕设/code/实验/GRU4Rec/mooc/item_knn_new.pkl', 'rb')
item_knn = pickle.load(f)
items_to_predict = train_data['ItemId'].unique()
if __name__ == '__main__':
    app.run()
