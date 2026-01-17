import math

# key=courseName,val=courseId
course_dic = {}
with open('course_dic.csv', 'r') as f:
    load_dic = f.readlines()
for line in load_dic:
    line = line.split('\n')[0]
    key, val = line.split(',')
    course_dic[key] = val

# generate course feature by concept
# key=conceptName,val=conceptId e.g.'K_活性炭_化学': 0
concept_dic = {}
# key=courseName,val=conceptNameList
course_concept = {}
# key=conceptId,val=appear times of concept in courseSet e.g. 0: 6, 1: 17
concept_appear_times = {}

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
        concept_appear_times[i] = 1
        i += 1
    else:
        concept_appear_times[concept_dic[value]] += 1
print(len(course_concept))
print("concept:",len(concept_dic))
print(concept_appear_times)

# print(course_concept.items())
num_of_documents = len(course_concept)

# key=conceptId,val=concept_tf-idf
weight = {}
for concept_id, count in concept_appear_times.items():
    weight[concept_id] = 1 * math.ceil(math.log(num_of_documents / count, 10))
    # weight[concept_id] = 1 * num_of_documents / count
print(weight)


def takeVal(elem):
    return concept_dic[elem]

with open("course_tf-idf.lsvm", 'w')as f2:
    for key, value_list in course_concept.items():
        line = str(course_dic[key])
        value_list.sort(key=takeVal)
        for val in value_list:
            # line += " %d:%d" % (concept_dic[val], weight[concept_dic[val]])
            line += " %d:%d" % (concept_dic[val], weight[concept_dic[val]])

        line += '\n'
        f2.write(line)
weight = 1
with open("course.lsvm", 'w')as f2:
    for key, value_list in course_concept.items():
        line = str(course_dic[key])
        value_list.sort(key=takeVal)
        for val in value_list:
            line += " %d:%d" % (concept_dic[val], weight)
        line += '\n'
        f2.write(line)


# # # generate course feature by video
# video_dic = {}
# # key=courseName,val=videoNameList
# course_video = {}
# # key=videoId,val=appear times of video in courseSet e.g. 0: 6, 1: 17
# video_appear_times = {}
#
# # 读course_video
# i = 0
# with open("course-video.json", 'r', encoding='utf-8') as f:
#     load_dict = f.readlines()
# for line in load_dict:
#     line = line.split('\n')[0]
#     [key, value] = line.split('\t')
#     if key not in course_video.keys():
#         course_video[key] = []
#     course_video[key].append(value)
#     if value not in video_dic.keys():
#         video_dic[value] = i
#         video_appear_times[i] = 1
#         i += 1
#     else:
#         video_appear_times[video_dic[value]] += 1
# print(len(course_video))
# # print(video_dic)
# print(video_appear_times)
#
# num_of_documents = len(course_video)
#
# # key=videoId,val=video_tf-idf
# weight = {}
# for video_id, count in video_appear_times.items():
#     weight[video_id] = 1 * math.ceil(math.log(num_of_documents / count, 10))
#     # weight[concept_id] = 1 * num_of_documents / count
# print(weight)
#
#
# def takeVal(elem):
#     return video_dic[elem]
#
#
# with open("course_tf-idf_v.lsvm", 'w')as f2:
#     for key, value_list in course_video.items():
#         line = str(course_dic[key])
#         value_list.sort(key=takeVal)
#         for val in value_list:
#             line += " %d:%d" % (
#                 video_dic[val], 1)
#         line += '\n'
#         f2.write(line)

# with open("concept_dic.csv", 'w')as f1:
#     for key, val in concept_dic.items():
#         line = key + "," + str(val) + "\n"
#         f1.write(line)


