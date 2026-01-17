import json
import pymysql

from config_db import connection

cursor = connection.cursor()
query_sql = 'select cid,cname from course'
cursor.execute(query_sql)
query_result = cursor.fetchall()
course_dic = {}
# 读db保存到course_dic,其中key=cid,value=cname
for item in query_result:
    course_dic[item[0]] = item[1]
count = 0
insert_sql = 'insert into course_prerequisites(pre_cid,pre_cname,suc_cid,suc_cname) values(%s,%s,%s,%s)'
with open('/Users/dmrfcoder/Documents/毕设/code/实验/mooccube_process/anno_course_prerequisite.json') as f:
    line = f.readline()
    while line:
        line = json.loads(line)
        for pre_cid in line['prerequisites']:
            pre_cname = course_dic[pre_cid]
            suc_cid = line['id']
            suc_cname = line['name']
            try:
                cursor.execute(insert_sql, (pre_cid, pre_cname, suc_cid, suc_cname))
                count += 1
                print((pre_cid, pre_cname, suc_cid, suc_cname))
            except pymysql.err.IntegrityError as err:
                print(err)
        line = f.readline()

connection.commit()
connection.close()
print(count)
print("Done")
