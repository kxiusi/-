import json
import pymysql

from config_db import connection

sql = 'insert into USER(uid,uname,upassword)VALUES (%s,%s,%s)'
cur = connection.cursor()
count = 0
with open('/Users/dmrfcoder/Documents/毕设/data/MOOCCube/entities/user.json') as f:
    line = f.readline()
    while line:
        line = json.loads(line)
        uid = line['id']
        uname = line['name']
        upassword = '123456'
        try:
            cur.execute(sql, (uid, uname, upassword))
            count += 1
        except pymysql.err.IntegrityError as err:
            print(err)
        finally:
            line = f.readline()
connection.commit()
connection.close()
print(count)
print("Done")
