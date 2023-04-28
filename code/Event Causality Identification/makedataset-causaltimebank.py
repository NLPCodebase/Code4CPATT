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
# cls_prompt = ""
cls_prompt = " causal </s></s> unrelated </s></s> "

Source = []
Answer = []
event1 = []
event2 = []
label = []


dataset_dir = os.path.join(
    "./data/Causal-TimeBank-main/Causal-TimeBank-CAT"
)
data_list = os.listdir(dataset_dir)
for data in data_list:

    
    annotated_data = etree.parse(
        "./data/Causal-TimeBank-main/Causal-TimeBank-CAT" + "/" + data,
        etree.XMLParser(remove_blank_text=True),
    )
    # result = annotated_data.xpath('//token[@t_id="2"]//text()')
    root_annotated_data = annotated_data.getroot()
    
    action_dict = collections.defaultdict(list)

    for elem in root_annotated_data.findall("Markables/"):
        for token_id in elem.findall("token_anchor"):
            if elem.tag.startswith("EVENT"):
                event_mention_id = elem.get("id", "nothing")
                token_mention_id = token_id.get("id", "nothing")
                action_dict[event_mention_id].append(token_mention_id)
    
    Relation = []
    counter = 0
    for elem in root_annotated_data.findall("Relations/CLINK/"):
        # counter = 0 , source ; counter = 1 , target

        if elem.tag.startswith("source"):
            event_mention_id_source = elem.get("id", "nothing")
        if elem.tag.startswith("target"):
            event_mention_id_target = elem.get("id", "nothing")
        if counter == 0:
            counter += 1
        else:
            counter = 0
            Relation.append(event_mention_id_source +
                            "\t"+event_mention_id_target)

    for event_pair in Relation:
        event1_str = ""
        event2_str = ""
        sentence1_str = ""
        sentence2_str = ""
        # event1
        token_list1 = action_dict[event_pair.split("\t")[0]]
        for token in token_list1:
            event1_str += (
                str(
                    annotated_data.xpath(
                        '//token[@id="' + token + '"]//text()'
                    )
                )
                + " "
            )
            sentence1_id = str(
                annotated_data.xpath(
                    '//token[@id="' + token + '"]//@sentence'
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
        token_list2 = action_dict[event_pair.split("\t")[1]]
        for token in token_list2:
            event2_str += (
                str(
                    annotated_data.xpath(
                        '//token[@id="' + token + '"]//text()'
                    )
                )
                + " "
            )
            sentence2_id = str(
                annotated_data.xpath(
                    '//token[@id="' + token + '"]//@sentence'
                )
            )

        for c in string.punctuation:
            event2_str = event2_str.replace(c, "")
            sentence2_id = sentence2_id.replace(c, "")

        for word in annotated_data.xpath(
            '//token[@sentence="' + sentence2_id + '"]//text()'
        ):
            sentence2_str += word + " "

        
        event1.append(event1_str)
        event2.append(event2_str)
        Answer.append(event1_str + "is the cause of " + event2_str)
        label.append("Cause-Effect")
        if sentence1_str == sentence2_str:
            Source.append(cls_prompt + sentence1_str)
        else:
            Source.append(cls_prompt + sentence1_str + sentence2_str)

    

    for i in range(max(1,len(Relation))):
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
                            '//token[@id="' + token + '"]//text()'
                        )
                    )
                    + " "
                )
                sentence1_id = str(
                    annotated_data.xpath(
                        '//token[@id="' + token + '"]//@sentence'
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
                            '//token[@id="' + token + '"]//text()'
                        )
                    )
                    + " "
                )
                sentence2_id = str(
                    annotated_data.xpath(
                        '//token[@id="' + token + '"]//@sentence'
                    )
                )

            for c in string.punctuation:
                event2_str = event2_str.replace(c, "")
                sentence2_id = sentence2_id.replace(c, "")

            for word in annotated_data.xpath(
                '//token[@sentence="' + sentence2_id + '"]//text()'
            ):
                sentence2_str += word + " "

            

            event1.append(event1_str)
            event2.append(event2_str)
            Answer.append(
                event1_str +  "has no relation to " + event2_str 
            )
            label.append("NONE")
            if sentence1_str == sentence2_str:
                Source.append(cls_prompt + sentence1_str)
            else:
                Source.append(cls_prompt + sentence1_str + sentence2_str)


dataframe = pd.DataFrame(
    {
        "Source sentence": Source,
        "Answer sentence": Answer,
        "event1": event1,
        "event2": event2,
        "label": label,
    }
)


dataframe.to_csv("./data/event_causal_timebank.csv",
                 index=False, sep=",", encoding="UTF-8")
