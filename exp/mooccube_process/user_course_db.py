import json
import pymysql

from config_db import connection

cursor = connection.cursor()
count = 0
insert_sql = 'insert into user_course(uid,cid) values(%s,%s)'
with open('/Users/dmrfcoder/Documents/毕设/code/实验/mooccube_process/user-course.json') as f:
    lines = f.readlines()
for line in lines:
    line = line.split('\n')[0]
    uid, cid = line.split('	')
    try:
        cursor.execute(insert_sql, (uid, cid))
        count += 1
    except pymysql.err.IntegrityError as err:
        print(err)
connection.commit()
connection.close()
print(count)
print("Done")
