import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mooc_cube', help='diginetica/Tmall/Nowplaying/mooc_cube')
parser.add_argument('--sample_num', type=int, default=12)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num

seq = pickle.load(open('datasets/' + dataset + '/all_train_seq.txt', 'rb'))

if dataset == 'diginetica':
    num = 43098
elif dataset == "Tmall":
    num = 40728
elif dataset == "Nowplaying":
    num = 60417
elif dataset == "mooc_cube":
    num = 681
elif dataset == "mooc_cube_video":
    num = 17466 
else:
    num = 3

relation = [[] for _ in range(2)]
neighbor = [] * num

all_test = set()

adj1 = [dict() for _ in range(num)]
adj2 = [dict() for _ in range(num)]
adj = [[] for _ in range(num)]

for i in range(len(seq)):
    data = seq[i]
    for k in range(1,3):

        for j in range(len(data)-k):
            relation[k-1].append([data[j], data[j+k]])
            relation[k-1].append([data[j+k], data[j]])

for tup in relation[0]:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1

for tup in relation[1]:
    if tup[1] in adj2[tup[0]].keys():
        adj2[tup[0]][tup[1]] += 1
    else:
        adj2[tup[0]][tup[1]] = 1

weight1 = [[] for _ in range(num)]
weight2 = [[] for _ in range(num)]

for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj1[t] = [v[0] for v in x]
    weight1[t] = [v[1] for v in x]

    x = [v for v in sorted(adj2[t].items(), reverse=True, key=lambda x: x[1])]
    adj2[t] = [v[0] for v in x]
    weight2[t] = [v[1] for v in x]

for i in range(num):
    adj1[i] = adj1[i][:sample_num]
    weight1[i] = weight1[i][:sample_num]
    adj2[i] = adj2[i][:sample_num]
    weight2[i] = weight2[i][:sample_num]

pickle.dump(adj, open('datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
# pickle.dump(weight, open('datasets/' + dataset + '/num_' + str(sample_num) + '.pkl', 'wb'))
