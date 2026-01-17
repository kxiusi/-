# 根据concept.json文件生成concept字典 key为concept-id val为从0开始的递增序号
import json

concept_dic = {}
# i = 0
# with open("concept.json", 'r', encoding='utf-8')as f1:
#     load_dic = f1.readlines()
# for line in load_dic:
#     concept_obj = json.loads(line)
#     concept_dic[concept_obj['id']] = i
#     line = line.split('\n')[0]
#     i += 1
course_concept = {}

course_dic = {}
with open('course_dic.csv', 'r') as f:
    load_dic = f.readlines()
for line in load_dic:
    line = line.split('\n')[0]
    key, val = line.split(',')
    course_dic[key] = val

# 读course_concept
i = 0
with open("course-concept.json", 'r', encoding='utf-8') as f:
    load_dict = f.readlines()
for line in load_dict:
    line = line.split('\n')[0]
    [key, value] = line.split('\t')
    if key not in course_concept.keys():
        course_concept[key] = []
    course_concept[key].append(value)
    if value not in concept_dic.keys():
        concept_dic[value] = i
        i += 1
print(len(concept_dic))
# print(course_concept.items())
weight = 1


def takeVal(elem):
    return concept_dic[elem]


with open("course_sparse.lsvm", 'w')as f2:
    feature = [0] * len(concept_dic)
    for key, value_list in course_concept.items():
        line = str(course_dic[key])
        value_list.sort(key=takeVal)
        for val in value_list:
            feature[concept_dic[val]] = 1
        line += ','.join([str(x) for x in feature])
        line += '\n'
        f2.write(line)

#
#         # f2.write(str(course_dic[key])+' ')
#         # for val in value_list:
#         #     f2.write(str(concept_dic[val])+':1 ')
#         # f2.write('\n')
# f2.close()
