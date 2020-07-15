#   Copyright 2020 trueto

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import re
import json
import chardet
import pandas as pd
from pathlib import Path

str_type = {
    "疾病和诊断": "Dis",
    "解剖部位": "Body",
    "实验室检验": "Test",
    "影像检查": "CT",
    "药物": "Drug",
    "手术": "Sur"
}

def C_trans_to_E(string):
    E_pun = u',.!?[]()<>"\'"\':;'
    C_pun = u'，。！？【】（）《》“‘”’：；'
    table= {ord(f): ord(t) for f, t in zip(C_pun, E_pun)}
    string = string.translate(table)
    return re.sub("[ |\t|\r|\n|\\\]", "_", string)

def strQ2B(ustr):
    "全角转半角"
    rstr = ""
    for uchar in ustr:
        inside_code = ord(uchar)
        # 全角空格直接转换
        if inside_code == 12288:
            inside_code = 32
        # 全角字符（除空格）根据关系转化
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstr += chr(inside_code)
    return rstr


def get_X_y(in_file, out_file, max_len=500):
    X = []
    y = []
    entity_data = []
    with open(in_file, 'r', encoding='utf8') as f:
        for line in f:
            tempObj = json.loads(line)
            originalText = tempObj['originalText']
            text = C_trans_to_E(strQ2B(originalText))
            entities = tempObj['entities']
            print("Processing text:{}".format(text))
            if len(text) <= max_len:
                X_ = list(text)
                y_ = ["O"] * len(X_)
                for entity in entities:
                    start_pos = entity["start_pos"]
                    end_pos = entity["end_pos"]
                    label_type = entity["label_type"]
                    if "clinical" in in_file:
                        tag = str_type[label_type]
                    else:
                        tag = label_type
                    # for i in range(start_pos, end_pos):
                    #    y_[i] = tag
                    entity_data.append([text[start_pos : end_pos], tag])
                    y_[start_pos] = 'B-'+tag
                    for i in range(start_pos+1, end_pos):
                        y_[i] = 'I-' + tag

                assert len(X_) == len(y_)

                X.append(X_)
                y.append(y_)
            else:
                # 分句
                dot_index_list = []
                text_ = text
                flag = 0
                while(len(text_) > max_len):
                    text_ = text_[:max_len]
                    index_list = []
                    for match in re.finditer(',', text_):
                        index = match.span()[0]
                        index_list.append(index)

                    # last_dot = index_list.pop()
                    if len(index_list) > 1:
                        last_dot = index_list.pop()
                    else:
                        index_list_ = []
                        for match in re.finditer('.', text_):
                            index = match.span()[0]
                            index_list_.append(index)

                        if len(index_list_) > 1:
                            last_dot = index_list_.pop()
                        else:
                            last_dot = len(text_)
                    dot_index_list.append(last_dot + flag)
                    text_ = text[last_dot+flag:]
                    flag += last_dot

                print(dot_index_list)
                flag = 0
                dot_index_list.append(len(text))
                for i, dot_index in enumerate(dot_index_list):
                    short_text = text[flag: dot_index+1]
                    X_ = list(short_text)
                    print("Short text:{}".format(short_text))
                    y_ = ["O"] * len(X_)
                    for entity in entities:
                        start_pos = entity["start_pos"]
                        end_pos = entity["end_pos"]
                        label_type = entity["label_type"]
                        if "clinical" in in_file:
                            tag = str_type[label_type]
                        else:
                            tag = label_type
                        #for j in range(start_pos, end_pos):
                        #    j = j - flag
                        #    if j >= 0 and j < len(y_):
                        #        y_[j] = tag
                        en_list = []

                        k = start_pos - flag
                        if k >= 0 and k < len(y_):
                            y_[k] = 'B-' + tag
                            en_list.append(X_[k])
                        for j in range(start_pos+1, end_pos):
                            j = j - flag
                            if j >= 0 and j < len(y_):
                                y_[j] = 'I-' + tag
                                en_list.append(X_[j])

                        if len(en_list) > 0:
                            entity_data.append(["".join(en_list), tag])
                    # if start_pos - flag > 0:
                    #    print(short_text[start_pos - flag : end_pos - flag])
                    assert len(X_) == len(y_)
                    X.append(X_)
                    y.append(y_)
                    flag = dot_index + 1

    assert len(X) == len(y)
    data_obj = (X, y, entity_data)
    pd.to_pickle(data_obj, out_file)

def get_X(in_file, out_file, max_len=500):
    X = []
    cut_his = {}
    originalTexts = []
    texts = []
    with open(in_file, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']

    with open(in_file, 'r', encoding="utf8") as f:
        for text_id, line in enumerate(f):
            tempObj = json.loads(line, encoding=encoding)
            originalText = tempObj['originalText']
            originalTexts.append(originalText)
            text = C_trans_to_E(strQ2B(originalText))
            texts.append(text)
            print("Processing text:{}".format(text))
            if len(text) <= max_len:
                X_ = list(text)
                X.append(X_)
                cut_his[text_id] = len(X) - 1
            else:
                # 分句
                dot_index_list = []
                text_ = text
                flag = 0
                while(len(text_) > max_len):
                    text_ = text_[:max_len]
                    index_list = []
                    for match in re.finditer(',', text_):
                        index = match.span()[0]
                        index_list.append(index)

                    # last_dot = index_list.pop()
                    if len(index_list) > 1:
                        last_dot = index_list.pop()
                    else:
                        index_list_ = []
                        for match in re.finditer('.', text_):
                            index = match.span()[0]
                            index_list_.append(index)

                        if len(index_list_) > 1:
                            last_dot = index_list_.pop()
                        else:
                            last_dot = len(text_)

                    dot_index_list.append(last_dot + flag)
                    text_ = text[last_dot+flag:]
                    flag += last_dot

                print(dot_index_list)
                flag = 0
                dot_index_list.append(len(text))
                text_id_list = []
                for i, dot_index in enumerate(dot_index_list):
                    short_text = text[flag: dot_index+1]
                    X_ = list(short_text)
                    X.append(X_)
                    text_id_list.append(len(X)-1)
                    flag = dot_index + 1

                cut_his[text_id] = text_id_list

    # assert len(X) == len(ids)
    data_obj = (X, cut_his, originalTexts, texts)
    pd.to_pickle(data_obj, out_file)

def get_vocab_csv(input_file, name):
    _, _, entity_data = pd.read_pickle(input_file)
    tmp_df = pd.DataFrame(data=entity_data, columns=['entity', 'label_type'])
    tmp_df.drop_duplicates(inplace=True)
    for label_type, entity_df in tmp_df.groupby(by='label_type', sort=False):
        entity_df.to_csv("data/{}_{}_vocab.csv".format(name, label_type), index=None)

if __name__ == '__main__':
    # get_X_y("data/clinical_ner_train.txt", "data/clinical_ner_train.pkl", max_len=125)
    # get_X("data/clinical_ner_dev.txt", "data/clinical_ner_dev.pkl", max_len=125)
    # get_X("data/clinical_ner_test.txt", "data/clinical_ner_test.pkl", max_len=125)
    # get_vocab_csv("data/clinical_ner_train.pkl", name="clinical")

    # get_X_y("data/diabetes_ner_train.txt", "data/diabetes_ner_train.pkl", max_len=125)
    get_X("data/diabetes_ner_dev.txt", "data/diabetes_ner_dev.pkl", max_len=125)
    get_X("data/diabetes_ner_test.txt", "data/diabetes_ner_test.pkl", max_len=125)
    # get_vocab_csv("data/diabetes_ner_train.pkl", name="diabetes")
