import json

import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.size'] = 10  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号



# 用于显示图片
def ShowGraph(G):
    # 使用matplotlib保存图片
    plt.figure(figsize=(11,8),frameon=False)  # 这里控制画布的大小，可以说改变整张图的布局

    pos = nx.spring_layout(G,k=2)

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)

    #nx.draw_networkx_edges(G, pos, edgelist=[("a", "e")], edge_color='r', arrows=True)
    #nx.draw(G, pos, labels=labeldict, with_labels=True,arrows=True)
    # node_labels = nx.get_node_attributes(G, 'name')  # 获取节点的desc属性
    # nx.draw_networkx_labels(G, pos, node_labels=node_labels, font_size=20)

    plt.savefig('数据prerequisites1.pdf')
    plt.show()


labeldict = {}
nodes = {}
G = nx.DiGraph()
with open('anno_course_prerequisite.json', 'r', encoding='utf=8') as f:
    load_dict = f.readlines()

for line in load_dict:
    line = line.split('\n')[0]
    obj = json.loads(line)
    # #以下用于画所有节点
    # G.add_node(obj['id'])
    # if obj['prerequisites']:
    #     for pre in obj['prerequisites']:
    #         G.add_edge(pre, obj['id'])
    #         labeldict[obj['id']] = obj['name']

    #以下用于画有边的节点
    nodes[obj['id']] = (obj['name'],obj['prerequisites'])
    labeldict[obj['id']]=obj['name'].split("（")[0]
edge_num=0
#edges=[]
for key,value in nodes.items():

    if value[1]:
        for pre in value[1]:
            if "数据" in labeldict[pre] or "数据" in labeldict[key] or "语言" in labeldict[pre] or "语言" in labeldict[key]:
                if labeldict[pre]==labeldict[key]:
                    continue
                G.add_edge(labeldict[pre], labeldict[key])
            edge_num+=1

ShowGraph(G)

print(edge_num)


