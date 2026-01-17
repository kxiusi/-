# < Ai ,Aj >= 1 .(exsit cy ) ∧ (< cx , cy >= 1) ∧ (< cy, cz >= 1),cx ∈ CAi , cy ∈ CAi∩CAj , cz ∈ CAj .

import json

course_dic = {}
course_list=[]
# 保存key-value为概念-课程的关系，每一项即概念对应的所有课程列表concept1:[course1,course2...]
concept_course = {}
course_concept = {}
# 保存概念间pre关系，每一项key:pre_concept val:suc_concept_list:[concept1,concept2...]
concept_pres = {}
# 生成的课程间pre关系，每一项[pre_course,suc_course]
course_pres = {}
with open('course.json', 'r', encoding='utf-8') as f:
    load_dict = f.readlines()
for line in load_dict:
    line = line.split('\n')[0]
    line = json.loads(line)
    course_dic[line['id']] = line['name']


# 读文件写到course_concept中
with open("course-concept.json", 'r', encoding='utf-8') as f:
    load_dict = f.readlines()
for line in load_dict:
    line = line.split('\n')[0]
    [value, key] = line.split('\t')
    if key not in concept_course.keys():
        concept_course[key] = []
    concept_course[key].append(value)
    if value not in course_concept.keys():
        course_concept[value] = []
        course_list.append(value)
    course_concept[value].append(key)
print(course_list)
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

# 保存课程自身包含的所有概念间的先序关系对，即该course中的所有< cx , cy >= 1
# key为courseid，val为概念关系对列表[(c1,c2),(c3,c4),...]
course_inner_pre_con = {}
course_inner_suc_con = {}
# 遍历所有课
for course in course_concept:
    course_inner_pre_con[course] = []
    course_inner_suc_con[course] = []
    # 对于该课程中所有概念
    for concept in course_concept[course]:
        if concept in concept_pres.keys():
            for suc_concept in concept_pres[concept]:
                if suc_concept in course_concept[course]:
                    course_inner_pre_con[course].append(concept)
                    course_inner_suc_con[course].append(suc_concept)
#     if course_inner_pre_con[course]:
#         print(course, course_inner_pre_con[course], course_inner_suc_con[course])
#

count=0
for i in range(len(course_list)):
    for j in range(len(course_list)):
        if j==i and j<len(course_list):
            j+=1
            continue
        # if course_list[j]=='C_course-v1:XJTU+2018121301X+2018_T2':
        #     print(course_inner_pre_con[course_list[j]])
        #     print(course_inner_suc_con[course_list[j]])
        andset=set(course_inner_suc_con[course_list[i]])&set(course_inner_pre_con[course_list[j]])
        # if andset!=set():
        if len(andset)>12:
            if course_list[i] not in course_pres.keys():
                course_pres[course_list[i]]=[]
            # 去除value列表中的重复元素
            if course_list[j] not in course_pres[course_list[i]]:
                course_pres[course_list[i]].append(course_list[j])
                print(course_dic[course_list[i]],course_dic[course_list[j]])
                count+=1
print(count)
# #对于每一条pre关系[pre_concept,suc_concept]
# for item in concept_pres:
#     # 判断concepts是否在映射字典中出现
#     if item[0] in concept_course.keys() and item[1] in concept_course.keys():
#         #包含pre_concept的所有课程，即course_concept[item[0]]都是包含suc_concept的课程的prerequisite
#         for pre_course in concept_course[item[0]]:
#             for suc_course in concept_course[item[1]]:
#                 # 去除pre=suc的情况
#                 if pre_course!=suc_course:
#                     if pre_course not in course_pres.keys():
#                         course_pres[pre_course]=[]
#                     # 去除value列表中的重复元素
#                     if suc_course not in course_pres[pre_course]:
#                         # 交集不为空
#                         if set(course_concept[pre_course])&set(course_concept[suc_course])!=set():
#                             course_pres[pre_course].append(suc_course)
#
# i=0
with open("generated_course_pre_k=12.json",'w',encoding='utf-8') as f:
    for key,valuelist in  course_pres.items():
        for value in valuelist:
            #f.write(course_dic[key]+'\t'+course_dic[value]+'\n')
            f.write(key + '\t' + value + '\n')

