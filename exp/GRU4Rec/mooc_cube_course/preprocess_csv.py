import csv
from collections import defaultdict

id_to_itemId_dic = defaultdict(int)
with open('/Users/dmrfcoder/Documents/毕设/code/实验/mooccube_process/mooc_cube_dic.csv') as f:
    reader = csv.DictReader(f, fieldnames=['id', 'itemId'], delimiter=',')
    for item in reader:
        id_to_itemId_dic[item['id']] = int(item['itemId'])
with open('mooc_cube.csv', 'r') as f1, open('mooc_cube_new.csv', 'w') as f2:
    reader = csv.DictReader(f1, fieldnames=['sessionId', 'itemId'], delimiter=',')
    for item in reader:
        item['itemId'] = id_to_itemId_dic[item['itemId']]
        line = item['sessionId'] + ',' + str(item['itemId']) + '\n'
        f2.write(line)
