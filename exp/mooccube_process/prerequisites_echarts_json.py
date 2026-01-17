import json
from config_db import connection

# 查询课程节点
query_sql = 'select cid,cname from course'
cur = connection.cursor()
cur.execute(query_sql)
course_list = cur.fetchall()

# 查询pre边
query_sql = 'select pre_cid,suc_cid from course_prerequisites'
cur.execute(query_sql)
prerequisites_list = cur.fetchall()

graph = {"nodes": [], "links": [], "categories": []}

# category对应大学名称的list
university_list = []
university_category_count = 0
for course_tuple in course_list:
    university = course_tuple[0].split(":")[1].split("+")[0]
    if university not in university_list:
        university_list.append(university)
print(university_list)

# 写json
with open('prerequisites_graph.json', 'w', encoding='utf-8') as f:
    index = 1
    # 写nodes
    for course_tuple in course_list:
        university = course_tuple[0].split(":")[1].split("+")[0]
        category = university_list.index(university)
        name = course_tuple[1].split('(')[0]
        name = name.split('（')[0]
        graph["nodes"].append({
            "id": course_tuple[0],
            "name": name,
            "value": course_tuple[0],
            "category": category
        })
        index += 1
    # 写links
    for pre_tuple in prerequisites_list:
        graph["links"].append({
            "source": pre_tuple[0],
            "target": pre_tuple[1]
        })
    # 写categories
    for i in range(len(university_list)):
        graph["categories"].append({
            'name': university_list[i]
        })

    write_json = json.dumps(graph, ensure_ascii=False)
    f.write(write_json)

connection.close()
