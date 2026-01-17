from app import *
seq = [326, 100, 97, 110, 599, 25]
res1 = itemknn_predict(seq)
res2 = predict(seq)
print(res1['results'])
print(res2['results'])
