import json

with open('a_entmax.txt') as rf:
    lines = rf.readlines()
line_list = []
for i in range(len(lines)):
    line = lines[i]
    line = line.split('\n')[0]
    number_list = line.split(' ')
    col_list = []
    for j in range(len(number_list)):
        number = number_list[j]
        obj = {}
        obj['0'] = i
        obj['1'] = j
        obj['2'] = number
        col_list.append(obj)
    line_list.append(col_list)
with open('a_entmax.json', 'w') as wf:
    wf.write(json.dumps({"data": line_list}))
