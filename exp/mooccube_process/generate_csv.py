import json


def generate_course_csv():
    course = {}
    user_course = []
    with open('course.json', 'r', encoding='utf-8') as f:
        load_dict = f.readlines()
    i = 0
    for line in load_dict:
        line = line.split('\n')[0]
        line = json.loads(line)
        course[line['id']] = i
        i += 1
    print(course)
    # 生成课程字典 key为id val为从零开始的递增序号
    with open('course_dic.csv', 'w')as f:
        for key, val in course.items():
            f.write(key + ',' + str(val) + '\n')
    cur_uid = 'U_7001215'
    session_id = 0
    with open('user-course.json', 'r', encoding='utf-8') as f:
        load_dict = f.readlines()
    for line in load_dict:
        line = line.split('\n')[0]
        [uid, cid] = line.split('\t')
        if uid != cur_uid:
            session_id += 1
        cur_uid = uid
        user_course.append((session_id, course[cid]))
    with open('mooc_cube.csv', 'w')as f:
        for item in user_course:
            f.write(str(item[0]) + ',' + str(item[1]) + '\n')


def generate_video_csv():
    video = {}
    user_video = []
    with open('video.json', 'r', encoding='utf-8') as f:
        load_dict = f.readlines()
    i = 0
    for line in load_dict:
        line = line.split('\n')[0]
        line = json.loads(line)
        video[line['id']] = i
        i += 1
    print(video)
    # 生成video字典 key为id val为从零开始的递增序号
    with open('video_dic.csv', 'w')as f:
        for key, val in video.items():
            f.write(key + ',' + str(val) + '\n')
    cur_uid = 'U_7001215'
    session_id = 0
    with open('user-video.json', 'r', encoding='utf-8') as f:
        load_dict = f.readlines()
    for line in load_dict:
        line = line.split('\n')[0]
        [uid, vid] = line.split('\t')
        if uid != cur_uid:
            session_id += 1
        cur_uid = uid
        user_video.append((session_id, video[vid]))
    with open('mooc_cube_video.csv', 'w')as f:
        for item in user_video:
            f.write(str(item[0]) + ',' + str(item[1]) + '\n')

#
# generate_video_csv()


def generate_video_csv_last10():
    video = {}
    user_video = {}
    with open('video.json', 'r', encoding='utf-8') as f:
        load_dict = f.readlines()
    i = 0
    for line in load_dict:
        line = line.split('\n')[0]
        line = json.loads(line)
        video[line['id']] = i
        i += 1
    # 生成video字典 key为id val为从零开始的递增序号
    with open('video_dic.csv', 'w')as f:
        for key, val in video.items():
            f.write(key + ',' + str(val) + '\n')
    cur_uid = 'U_7001215'
    session_id = 0
    user_video[session_id] = []
    with open('user-video.json', 'r', encoding='utf-8') as f:
        load_dict = f.readlines()
    for line in load_dict:
        line = line.split('\n')[0]
        [uid, vid] = line.split('\t')
        if uid != cur_uid:
            session_id += 1
            user_video[session_id] = []
        cur_uid = uid
        if len(user_video[session_id]) >= 10:
            del user_video[session_id][0]
        user_video[session_id].append(video[vid])
    with open('mooc_cube_video_10.csv', 'w')as f:
        for key, val_list in user_video.items():
            for val in val_list:
                f.write(str(key) + ',' + str(val) + '\n')


#generate_video_csv_last10()
generate_course_csv()