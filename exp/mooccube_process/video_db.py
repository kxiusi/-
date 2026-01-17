import json

from dotenv import load_dotenv
import pymysql
import os

load_dotenv()
connection = pymysql.connect(
    host="localhost",
    database="mooc",
    user="root",
    password="xie.159753.",
)
cursor = connection.cursor()
sql = 'insert into video(vid,vname) values(%s,%s)'
count=0
with open('/Users/dmrfcoder/Documents/毕设/data/MOOCCube/entities/video.json') as f:
    line = f.readline()
    while line:
        dic = json.loads(line)
        vid = dic['id']
        vname = dic['name']
        try:
            cursor.execute(sql, (vid, vname))
            count+=1
        except pymysql.err.IntegrityError as err:
            print(err)
        finally:
            line = f.readline()
connection.commit()
connection.close()

print(count)
print("Done")
