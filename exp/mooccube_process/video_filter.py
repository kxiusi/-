import json
with open('E:\\COPY-JoeyMac2018\\文稿\\毕设\\data\\MOOCCube\\additional_information\\user_video_act.json','r')as f:
    line=f.readline()
    print(line)
    line=json.loads(line)
    print(len(line["activity"]))