# 根据concept.json文件生成concept字典 key为concept-id val为从0开始的递增序号
import json



video_dic = {}
with open('video_dic.csv', 'r') as f:
    load_dic = f.readlines()
for line in load_dic:
    line = line.split('\n')[0]
    key, val = line.split(',')
    video_dic[key] = val
concept_dic = {}
with open("concept_dic.csv",'r')as f1:
    load_dict = f1.readlines()
    for line in load_dict:
        line = line.split('\n')[0]

        line = line.split(',')
        if len(line)==2:
            concept_dic[line[0]]=int(line[1])
print(concept_dic)

# 读video_concept
i = 0
video_concept = {}
with open("video-concept.json", 'r', encoding='utf-8') as f:
    load_dict = f.readlines()
for line in load_dict:
    line = line.split('\n')[0]
    [key, value] = line.split('\t')
    if key in video_dic.keys() and value in concept_dic.keys():
        if key not in video_concept.keys():
            video_concept[key] = []
        video_concept[key].append(value)

# print(course_concept.items())
weight = 1


def takeVal(elem):
    return concept_dic[elem]

#
with open("video.lsvm", 'w')as f2:
    for key, value_list in video_concept.items():
        line = str(video_dic[key])
        value_list.sort(key=takeVal)
        for val in value_list:
            line += " %d:%d" % (concept_dic[val], weight)
        line += '\n'
        f2.write(line)
#
#         # f2.write(str(course_dic[key])+' ')
#         # for val in value_list:
#         #     f2.write(str(concept_dic[val])+':1 ')
#         # f2.write('\n')
# f2.close()
