from pickle import TRUE
import sys
import os
import os.path
from lxml import etree
import collections
import json
from itertools import combinations
import pandas as pd
import random
import string


random.seed(3407)
folder_dir = os.path.join("./data/EventStoryLine-master/annotated_data/v1.0")
folder_list = os.listdir(folder_dir)
folder_list.sort(key=lambda x: x.split(".")[0])
num = 0
cls_prompt = ""
# cls_prompt = " unrelated </s></s> causal </s></s> "
# cls_prompt = " causal </s></s> unrelated </s></s> "
# cls_prompt = " relevant </s></s> irrelevant </s></s> "
Source = []
Answer = []

event1 = []
event2 = []
label = []
type = []
for folder in folder_list:
    dataset_dir = os.path.join(
        "./data/EventStoryLine-master/annotated_data/v1.0/" + folder
    )
    data_list = os.listdir(dataset_dir)
    for data in data_list:
        if data[len(data) - 4 :] != ".xml":
            continue
        flag = False
        # 读标注数据
        annotated_data = etree.parse(
            "./data/EventStoryLine-master/annotated_data/v1.0/" + folder + "/" + data,
            etree.XMLParser(remove_blank_text=True),
        )
        # result = annotated_data.xpath('//token[@t_id="2"]//text()')
        root_annotated_data = annotated_data.getroot()
        # 检查是否有plot_link
        for elem in root_annotated_data.findall("Relations/"):
            if elem.tag == "PLOT_LINK":
                flag = TRUE
        # 读关系数据
        if flag:

            f = open(
                "./data/EventStoryLine-master/evaluation_format/full_corpus/v1.0/event_mentions_extended/"
                + folder
                + "/"
                + data[: len(data) - 4],
                "r",
            )
            Relation = f.readlines()  # 直接将文件中按行读到list里
            # 添加有因果关系的事件对
            for event_pair in Relation:
                event1_str = ""
                event2_str = ""
                sentence1_str = ""
                sentence2_str = ""
                # event1
                token_list1 = event_pair.split("\t")[0].split("_")
                for token in token_list1:
                    event1_str += (
                        str(
                            annotated_data.xpath(
                                '//token[@t_id="' + token + '"]//text()'
                            )
                        )
                        + " "
                    )
                    sentence1_id = str(
                        annotated_data.xpath(
                            '//token[@t_id="' + token + '"]//@sentence'
                        )
                    )

                for c in string.punctuation:
                    event1_str = event1_str.replace(c, "")
                    sentence1_id = sentence1_id.replace(c, "")

                for word in annotated_data.xpath(
                    '//token[@sentence="' + sentence1_id + '"]//text()'
                ):
                    sentence1_str += word + " "

                # event2
                token_list2 = event_pair.split("\t")[1].split("_")
                for token in token_list2:
                    event2_str += (
                        str(
                            annotated_data.xpath(
                                '//token[@t_id="' + token + '"]//text()'
                            )
                        )
                        + " "
                    )
                    sentence2_id = str(
                        annotated_data.xpath(
                            '//token[@t_id="' + token + '"]//@sentence'
                        )
                    )

                for c in string.punctuation:
                    event2_str = event2_str.replace(c, "")
                    sentence2_id = sentence2_id.replace(c, "")

                for word in annotated_data.xpath(
                    '//token[@sentence="' + sentence2_id + '"]//text()'
                ):
                    sentence2_str += word + " "

                # 根据关系调整顺序
                if event_pair.split("\t")[2] == "PRECONDITION\n":
                    
                    event1.append(event1_str)
                    event2.append(event2_str)
                    Answer.append(event1_str + "is the cause of " + event2_str)
                    label.append("Cause-Effect")
                    if sentence1_str == sentence2_str:
                        Source.append(cls_prompt + sentence1_str)
                        type.append("intra")
                    else:
                        Source.append(cls_prompt + sentence1_str + sentence2_str)
                        type.append("inter")

                else:
                    
                    event1.append(event2_str)
                    event2.append(event1_str)
                    Answer.append(event2_str + "is the cause of " + event1_str)
                    label.append("Cause-Effect")
                    if sentence1_str == sentence2_str:
                        Source.append(cls_prompt + sentence2_str)
                        type.append("intra")
                    else:
                        Source.append(cls_prompt + sentence2_str + sentence1_str)
                        type.append("inter")

            # 添加没有关系的事件对

            # 选择所有action
            action_dict = collections.defaultdict(list)

            for elem in root_annotated_data.findall("Markables/"):
                for token_id in elem.findall("token_anchor"):
                    if elem.tag.startswith("ACTION") or elem.tag.startswith(
                        ("NEG_ACTION")
                    ):
                        event_mention_id = elem.get("m_id", "nothing")
                        token_mention_id = token_id.get("t_id", "nothing")
                        action_dict[event_mention_id].append(token_mention_id)

            # 随机选20个事件对,检测事件对是否在文件中存在

            for i in range(20):
                random_event_pair = random.sample(list(action_dict), 2)
                event1_token_id = "_".join(action_dict[random_event_pair[0]])
                event2_token_id = "_".join(action_dict[random_event_pair[1]])
                exist_flag = False
                for event_pair in Relation:
                    exist_event1 = event_pair.split("\t")[0]
                    exist_event2 = event_pair.split("\t")[1]
                    if (
                        exist_event1 == event1_token_id
                        and exist_event2 == event2_token_id
                    ) or (
                        exist_event1 == event2_token_id
                        and exist_event2 == event1_token_id
                    ):
                        exist_flag = True
                
                event1_str = ""
                event2_str = ""
                sentence1_str = ""
                sentence2_str = ""
                if exist_flag:
                    continue
                else:
                    # event1
                    for token in action_dict[random_event_pair[0]]:
                        event1_str += (
                            str(
                                annotated_data.xpath(
                                    '//token[@t_id="' + token + '"]//text()'
                                )
                            )
                            + " "
                        )
                        sentence1_id = str(
                            annotated_data.xpath(
                                '//token[@t_id="' + token + '"]//@sentence'
                            )
                        )

                    for c in string.punctuation:
                        event1_str = event1_str.replace(c, "")
                        sentence1_id = sentence1_id.replace(c, "")

                    for word in annotated_data.xpath(
                        '//token[@sentence="' + sentence1_id + '"]//text()'
                    ):
                        sentence1_str += word + " "
                    # event2
                    for token in action_dict[random_event_pair[1]]:
                        event2_str += (
                            str(
                                annotated_data.xpath(
                                    '//token[@t_id="' + token + '"]//text()'
                                )
                            )
                            + " "
                        )
                        sentence2_id = str(
                            annotated_data.xpath(
                                '//token[@t_id="' + token + '"]//@sentence'
                            )
                        )

                    for c in string.punctuation:
                        event2_str = event2_str.replace(c, "")
                        sentence2_id = sentence2_id.replace(c, "")

                    for word in annotated_data.xpath(
                        '//token[@sentence="' + sentence2_id + '"]//text()'
                    ):
                        sentence2_str += word + " "

                    # 加入文件
                   
                    event1.append(event1_str)
                    event2.append(event2_str)
                    # Answer.append(
                    #     event1_str + "and " + event2_str + "have no relationship"
                    # )
                    Answer.append(
                        event1_str + "has no relation to " + event2_str
                    )
                    label.append("NONE")
                    if sentence1_str == sentence2_str:
                        Source.append(cls_prompt + sentence1_str)
                        type.append("intra")
                    else:
                        Source.append(cls_prompt + sentence1_str + sentence2_str)
                        type.append("inter")

            f.close()  # 关闭文件


dataframe = pd.DataFrame(
    {
        "Source sentence": Source,
        "Answer sentence": Answer,
        "event1": event1,
        "event2": event2,
        "label": label,
        "type": type
    }
)

# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("./data/event_eventstoryline_simple.csv", index=False, sep=",", encoding="UTF-8")

