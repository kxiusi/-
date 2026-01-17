import csv
import json
import pymysql
from collections import defaultdict

connection = pymysql.connect(
    host="localhost",
    database="mooc",
    user="root",
    password="xie.159753.",
)


def insert_course_from_json():
    cursor = connection.cursor()
    sql = 'insert into course(cid,cname,prerequisites,about) values(%s,%s,%s,%s)'
    count = 0
    with open('/Users/dmrfcoder/Documents/毕设/code/实验/mooccube_process/course.json') as f:
        line = f.readline()
        while line:
            dic = json.loads(line)
            cid = dic['id']
            cname = dic['name']
            prerequisites = dic['prerequisites']
            about = dic['about']
            if len(about) > 1000:
                about = about[:995] + '……'
            try:
                cursor.execute(sql, (cid, cname, prerequisites, about))
                count += 1
            except pymysql.err.IntegrityError as err:
                print(err)
            finally:
                line = f.readline()
    connection.commit()
    print(count)
    print("Done")


def update_popular():
    popular_dic = defaultdict(int)
    with open('/Users/dmrfcoder/Documents/毕设/code/实验/mooccube_process/user-course.json') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split('\n')[0]
        cid = line.split('	')[1]
        popular_dic[cid] += 1
    cursor = connection.cursor()
    update_sql = 'update course set popular=%s where cid=%s'
    try:
        for cid, popular in popular_dic.items():
            cursor.execute(update_sql, (popular, cid))
    except pymysql.err.IntegrityError as err:
        print(err)
    connection.commit()
    print("Done")


def update_dic_id():
    update_sql = 'update course set itemId=%s where cid=%s'
    cursor = connection.cursor()
    count = 0
    id_to_itemId_dic = defaultdict(int)
    # twice mapping
    # 1. cid->id(0-680) in course_dic.csv
    # 2. id->itemid(used in dataset) in mooc_cube_dic.csv
    with open('/Users/dmrfcoder/Documents/毕设/code/实验/mooccube_process/mooc_cube_dic.csv') as f:
        reader = csv.DictReader(f, fieldnames=['id', 'itemid'], delimiter=',')
        for item in reader:
            id_to_itemId_dic[item['id']] = int(item['itemid'])
    with open('/Users/dmrfcoder/Documents/毕设/code/实验/mooccube_process/course_dic.csv') as f:
        reader = csv.DictReader(f, fieldnames=['cid', 'id'], delimiter=',')
        for item in reader:
            cid = item['cid']
            itemId = id_to_itemId_dic[item['id']]
            try:
                cursor.execute(update_sql, (itemId, cid))
                count += 1
            except pymysql.err.IntegrityError as err:
                print(err)
    connection.commit()
    print(count)
    print("Done")


# update_popular()
update_dic_id()
connection.close()
