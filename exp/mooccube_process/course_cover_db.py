import json
import requests
import pymysql
from collections import defaultdict

connection = pymysql.connect(
    host="localhost",
    database="mooc",
    user="root",
    password="xie.159753.",
)
cursor = connection.cursor()
sql = 'select cid,cname,cover from course'
cursor.execute(sql)
query_result = cursor.fetchall()
course_dic = defaultdict(list)
for item in query_result:
    name = item[1]
    cover = item[2]
    name = name.split('(')[0]
    name = name.split('（')[0]
    course_dic[name].append((item[0], item[2]))

url = "https://www.xuetangx.com/api/v1/lms/get_product_list/?page="

payload = "{\"query\":\"\",\"chief_org\":[\"1\"],\"classify\":[],\"selling_type\":[],\"status\":[],\"appid\":10000}"
headers = {
    'Host': 'www.xuetangx.com',
    'Cookie': 'provider=xuetang; django_language=zh; sajssdk_2015_cross_new_user=1; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%221843de8c8cb644-0ae7a10b72b0d9-18525635-1296000-1843de8c8ccc0b%22%2C%22first_id%22%3A%22%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E8%87%AA%E7%84%B6%E6%90%9C%E7%B4%A2%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC%22%2C%22%24latest_referrer%22%3A%22https%3A%2F%2Fwww.google.com.hk%2F%22%7D%2C%22%24device_id%22%3A%221843de8c8cb644-0ae7a10b72b0d9-18525635-1296000-1843de8c8ccc0b%22%7D; _abfpc=83f3a3e910ddf60b93207d86d5e5a0eccca3b1c0_2.0; _ga=GA1.2.671113110.1667485978; _gid=GA1.2.89168925.1667485978; point={%22point_active%22:true%2C%22platform_task_active%22:true%2C%22learn_task_active%22:true}; cna=bb9775fb6ac0d3c70ceb163e4a345314; JG_016f5b1907c3bc045f8f48de1_PV=1667485979107|1667485992273',
    'sec-ch-ua': '"Google Chrome";v="107", "Chromium";v="107", "Not=A?Brand";v="24"',
    'django-language': 'zh',
    'accept-language': 'zh',
    'sec-ch-ua-mobile': '?0',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
    'app-name': 'xtzx',
    'content-type': 'application/json',
    'accept': 'application/json, text/plain, */*',
    'terminal-type': 'web',
    'xtbz': 'xt',
    'x-client': 'web',
    'sec-ch-ua-platform': '"macOS"',
    'origin': 'https://www.xuetangx.com',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-mode': 'cors',
    'sec-fetch-dest': 'empty',
    'referer': 'https://www.xuetangx.com/search?query=&org=1&classify=&type=&status=&ss=manual_search&page=1'
}

sql = 'update course set cover=%s where cid=%s'
count = 0
# 爬到的cover存db
# for page_index in range(1, 46):
#     response = requests.request("POST", url + str(page_index), headers=headers, data=payload)
#     response_obj = json.loads(response.text)
#     for item_course in response_obj['data']['product_list']:
#         name = item_course['name']
#         cover = item_course['cover']
#         if course_dic[name]:
#             for cid in course_dic[name]:
#                 try:
#                     cursor.execute(sql, (cover, cid))
#                     count += 1
#                 except pymysql.err.IntegrityError as err:
#                     print(err)

# 查询未匹配到name的课程cover
base_payload = "{\"query\":\"%s\",\"chief_org\":[],\"classify\":[],\"selling_type\":[],\"status\":[],\"appid\":10000}"

for key, value in course_dic.items():
    for item in value:
        if item[1] is None:
            payload = base_payload % key
            response = requests.request("POST", url + '1', headers=headers, data=payload.encode("utf-8"))
            response_obj = json.loads(response.text)
            if response_obj['data']['count'] == 0:
                continue
            item_course = response_obj['data']['product_list'][0]
            name = item_course['name']
            cover = item_course['cover']
            try:
                cursor.execute(sql, (cover, item[0]))
                print(name + "..." + cover)
                count += 1
            except pymysql.err.IntegrityError as err:
                print(err)

connection.commit()
connection.close()

print(count)
print("Done")
