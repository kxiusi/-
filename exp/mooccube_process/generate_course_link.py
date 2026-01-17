import json

course_dic = {}
with open('course_dic.csv', 'r') as f:
    load_dic = f.readlines()
for line in load_dic:
    line = line.split('\n')[0]
    key, val = line.split(',')
    course_dic[key] = val
    # print(key,val)

# with open('anno_course_prerequisite.json', 'r', encoding='utf-8') as f1, open('course.link', 'w') as f2:
#     load_dict = f1.readlines()
#     for line in load_dict:
#         line = line.split('\n')[0]
#         obj = json.loads(line)
#         if obj['prerequisites']:
#             for pre in obj['prerequisites']:
#                 f2.write(course_dic[pre] + ' ' + course_dic[obj['id']] + '\n')
#                 # print(course_dic[obj['id']]+' '+course_dic[pre])

with open('generated_course_pre_k=12.json', 'r', encoding='utf-8') as f1, open('course_dense_12.link', 'w') as f2:
    load_dict = f1.readlines()
    for line in load_dict:
        line = line.split('\n')[0].split('\t')
        pre, suc = line[0], line[1]
        f2.write(course_dic[pre] + ' ' + course_dic[suc] + '\n')
