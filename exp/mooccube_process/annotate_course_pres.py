import json

import jieba.analyse

course_list = []
course_dic = {}
with open('course.json', 'r', encoding='utf=8') as f:
    load_dict = f.readlines()
for line in load_dict:
    line = line.split('\n')[0]
    obj = json.loads(line)
    # 去掉name里的2019、自主模式
    course_name = obj['name'].split('（')[0]
    course_name = course_name.split('(')[0]
    # course_name = obj['name'].split('（自')[0]
    # course_name = course_name.split('(自')[0]
    # course_name = course_name.split('（20')[0]
    # course_name = course_name.split('(20')[0]
    course_dic[course_name] = obj['id']
    item = {}
    item['id'] = obj['id']
    item['name'] = obj['name']
    if obj['prerequisites'].find("无") != -1 or obj['prerequisites'].find("没有") != -1:
        item['prerequisites'] = None
    else:
        item['prerequisites'] = obj['prerequisites']
    course_list.append(item)
with open('pre_dict.txt', 'w', encoding='utf=8') as f:
    for key in course_dic.keys():
        f.write(key + '\n')
jieba.load_userdict('pre_dict.txt')
count_course = 0
count_pres = 0
result_list = []
tmp_item = {}
for item in course_list:
    flag = False
    tmp_item = {}
    tmp_item['id'] = item['id']
    tmp_item['name'] = item['name']
    tmp_item['prerequisites'] = []
    if item['prerequisites']:
        # item_pre=jieba.lcut(item['prerequisites'], cut_all=False, HMM=True)
        print(item['prerequisites'])

        item_pre = jieba.analyse.extract_tags(item['prerequisites'], topK=5)
        print(item_pre)
        for pre in item_pre:
            if pre in course_dic.keys():
                flag = True
                tmp_item['prerequisites'].append(course_dic[pre])
                print(pre, course_dic[pre])
                count_pres += 1
        if flag:
            count_course += 1
        else:
            print('None')
        print('\n')
    result_list.append(tmp_item)
print(count_course)
print(count_pres)

# if item['prerequisites'] in course_dic.keys():
#     item['prerequisites']=course_dic[item['prerequisites']]
#     print(item)
#
with open("anno_course_prerequisite.json", 'w') as f:
    for item in result_list:
        f.write(json.dumps(item, ensure_ascii=False))
        f.write('\n')
