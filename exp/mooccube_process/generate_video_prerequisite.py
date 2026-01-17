# < Ai ,Aj >= 1 .(exsit cy ) ∧ (< cx , cy >= 1) ∧ (< cy, cz >= 1),cx ∈ CAi , cy ∈ CAi∩CAj , cz ∈ CAj .

import json

video_dic = {}
video_list=[]
# 保存key-value为概念-课程的关系，每一项即概念对应的所有课程列表concept1:[video1,video2...]
concept_video = {}
video_concept = {}
# 保存概念间pre关系，每一项key:pre_concept val:suc_concept_list:[concept1,concept2...]
concept_pres = {}
# 生成的课程间pre关系，每一项[pre_video,suc_video]
video_pres = {}
with open('video_dic.csv', 'r', encoding='utf-8') as f:
    load_dict = f.readlines()
for line in load_dict:
    line = line.split('\n')[0]
    key,val=line.split(',')
    video_dic[key] = val
# print(video_dic)

# 读文件写到video_concept中
with open("video-concept.json", 'r', encoding='utf-8') as f:
    load_dict = f.readlines()
for line in load_dict:
    line = line.split('\n')[0]
    [value, key] = line.split('\t')
    if key not in concept_video.keys():
        concept_video[key] = []
    concept_video[key].append(value)
    if value not in video_concept.keys():
        video_concept[value] = []
        video_list.append(value)
    video_concept[value].append(key)
# print(video_list)
# 读文件写到concept_pres中
with open("prerequisite-dependency.json", 'r', encoding='utf-8') as f:
    load_pres = f.readlines()
for line in load_pres:
    line = line.split('\n')[0]
    pre = line.split('	')[0]
    suc = line.split('	')[1]
    if pre not in concept_pres.keys():
        concept_pres[pre] = []
    concept_pres[pre].append(suc)

# 保存课程自身包含的所有概念间的先序关系对，即该video中的所有< cx , cy >= 1
# key为videoid，val为概念关系对列表[(c1,c2),(c3,c4),...]
video_inner_pre_con = {}
video_inner_suc_con = {}
# 遍历所有课
for video in video_concept:
    video_inner_pre_con[video] = []
    video_inner_suc_con[video] = []
    # 对于该课程中所有概念
    for concept in video_concept[video]:
        if concept in concept_pres.keys():
            for suc_concept in concept_pres[concept]:
                if suc_concept in video_concept[video]:
                    video_inner_pre_con[video].append(concept)
                    video_inner_suc_con[video].append(suc_concept)
#     if video_inner_pre_con[video]:
#         print(video, video_inner_pre_con[video], video_inner_suc_con[video])
#

count=0
for i in range(len(video_list)):
    for j in range(len(video_list)):
        if j==i and j<len(video_list):
            j+=1
            continue
        # if video_list[j]=='C_video-v1:XJTU+2018121301X+2018_T2':
        #     print(video_inner_pre_con[video_list[j]])
        #     print(video_inner_suc_con[video_list[j]])
        andset=set(video_inner_suc_con[video_list[i]])&set(video_inner_pre_con[video_list[j]])
        # if andset!=set():
        if len(andset)>12:
            if video_list[i] not in video_pres.keys():
                video_pres[video_list[i]]=[]
            # 去除value列表中的重复元素
            if video_list[j] not in video_pres[video_list[i]]:
                video_pres[video_list[i]].append(video_list[j])
                print(video_dic[video_list[i]],video_dic[video_list[j]])
                count+=1
print(count)
# #对于每一条pre关系[pre_concept,suc_concept]
# for item in concept_pres:
#     # 判断concepts是否在映射字典中出现
#     if item[0] in concept_video.keys() and item[1] in concept_video.keys():
#         #包含pre_concept的所有课程，即video_concept[item[0]]都是包含suc_concept的课程的prerequisite
#         for pre_video in concept_video[item[0]]:
#             for suc_video in concept_video[item[1]]:
#                 # 去除pre=suc的情况
#                 if pre_video!=suc_video:
#                     if pre_video not in video_pres.keys():
#                         video_pres[pre_video]=[]
#                     # 去除value列表中的重复元素
#                     if suc_video not in video_pres[pre_video]:
#                         # 交集不为空
#                         if set(video_concept[pre_video])&set(video_concept[suc_video])!=set():
#                             video_pres[pre_video].append(suc_video)
#
# i=0
with open("generated_video_pre_k=12.json",'w',encoding='utf-8') as f:
    for key,valuelist in  video_pres.items():
        for value in valuelist:
            #f.write(video_dic[key]+'\t'+video_dic[value]+'\n')
            f.write(key + '\t' + value + '\n')

