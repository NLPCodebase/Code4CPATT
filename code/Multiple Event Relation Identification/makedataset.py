import json
from itertools import combinations
import os
import pandas as pd
import random


dataset_dir = os.path.join('.', 'json')
data_list = os.listdir(dataset_dir)
data_list.sort(key=lambda x:x.split('.')[0])
Source=[]
Answer=[]
event1_list=[]
evemt2_list=[]
labels=[]
print(data_list)

print('Total news number:', len(data_list))

def find_all_path(graph, start, end, path=[]):
    """
    遍历图的所有路径
    :param graph:
    :param start:
    :param end:
    :param path: 存储路径
    :return:
    """
    path = path + [start]
    if start == end:
        return [path]
    num=0
    for node in graph[start]:
        if graph[start][node] == '并列关系':
            num=num+1
        if num == len(graph[start]):
            return [path]

    paths = []  # 存储所有路径

    for node in graph[start]:
        if graph[start][node] == '并列关系':
            continue
        if node not in path:
            
            newpaths = find_all_path(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)

    return paths

good_event_chain = []

for data_file_name in data_list:

        # 读取文件内容
        data_file_path = os.path.join(dataset_dir, data_file_name)
        with open(data_file_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())  # 文件JSON内容
        
        keys = sorted(data['event_relation'].keys())
        graph_start = keys[0]
        graph_end = data["event_element"][-1]["event_graph"][-1]["child_event_id"]

        # 获取第一句话的event node
        first_event_keys = [x["child_event_id"] for x in data["event_element"][0]["event_graph"]]

        # 获取所有完整事件链
        event_chains = []
        event_chains = find_all_path(data['event_relation'], graph_start, graph_end, event_chains)
       
        print(data_file_name)
        print(event_chains)

        # 事件链去重
        ret = []
        for event_item in event_chains:
            if event_item not in ret:
                ret.append(event_item)
        news_id = data_file_name.split('_')[0]
        ret_event_chains = ret
        del ret


        # 补充news id
        for event_item in ret_event_chains:
            good_event_chain.append([news_id + "_" + x for x in event_item])

        # 生成数据集
        for event_chain in event_chains:
            wholeevent=""
            for id in event_chain:
                for sentence in data["event_element"]:
                    for event in sentence["event_graph"]:
                        if id == event["child_event_id"]:
                            wholeevent+=event["trigger_subject"]+event["trigger"]+event["trigger_object"]+'。'
            #X列
            print(wholeevent)
            #Y列因果关系
            for start in event_chain:
                if start in data['event_relation']:
                    for node in data['event_relation'][start]:
                        if node in event_chain and data['event_relation'][start][node] == "因果关系":
                            for sentence in data["event_element"]:
                                for event in sentence["event_graph"]:
                                    if start == event["child_event_id"]:
                                        event1=event["trigger_subject"]+event["trigger"]+event["trigger_object"]
                            for sentence in data["event_element"]:
                                for event in sentence["event_graph"]:
                                    if node == event["child_event_id"]:
                                        event2=event["trigger_subject"]+event["trigger"]+event["trigger_object"]
                            Source.append(wholeevent)
                            Answer.append(event1+"是"+event2+"的原因事件。") 
                            event1_list.append(event1)
                            evemt2_list.append(event2)
                            labels.append("因果关系")

            #Y列顺承关系
                        if node in event_chain and data['event_relation'][start][node] == "顺承关系":
                            for sentence in data["event_element"]:
                                for event in sentence["event_graph"]:
                                    if start == event["child_event_id"]:
                                        event1=event["trigger_subject"]+event["trigger"]+event["trigger_object"]
                            for sentence in data["event_element"]:
                                for event in sentence["event_graph"]:
                                    if node == event["child_event_id"]:
                                        event2=event["trigger_subject"]+event["trigger"]+event["trigger_object"]
                            Answer.append(event2+"是"+event1+"的后续事件。") 
                            Source.append(wholeevent)
                            event1_list.append(event2)
                            evemt2_list.append(event1)
                            labels.append("顺承关系")
            
            

                    
            #Y列无关系
            if len(event_chain) > 4:
                i=0
                while i < 4:
                    start=event_chain[random.randint(0, len(event_chain)-1)]
                    end=event_chain[random.randint(0, len(event_chain)-1)]
                    if start!=end and start in data['event_relation'] and end in data['event_relation']:
                        if end not in data['event_relation'][start] and start not in data['event_relation'][end]:
                            for sentence in data["event_element"]:
                                for event in sentence["event_graph"]:
                                    if start == event["child_event_id"]:
                                        event1=event["trigger_subject"]+event["trigger"]+event["trigger_object"]
                            for sentence in data["event_element"]:
                                for event in sentence["event_graph"]:
                                    if end == event["child_event_id"]:
                                        event2=event["trigger_subject"]+event["trigger"]+event["trigger_object"]
                            Answer.append(event1+"是"+event2+"的无关事件。") 
                            Source.append(wholeevent)
                            event1_list.append(event1)
                            evemt2_list.append(event2)
                            labels.append("NONE")
                            i+=1

print('Total event chain number:', len(good_event_chain))
#字典中的key值即为csv中列名
print(len(Source))
print(len(Answer))
dataframe = pd.DataFrame({'Source sentence':Source,'Answer sentence':Answer,'Event1':event1_list,'Event2':evemt2_list,'labels':labels})

dataframe.to_csv("./data/event_train.csv",index=False,sep=',',encoding = 'gbk')