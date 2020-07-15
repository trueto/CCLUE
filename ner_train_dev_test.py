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
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from bertology_sklearn import BertologyTokenClassifier

str_type = {
    "疾病和诊断": "Dis",
    "解剖部位": "Body",
    "实验室检验": "Test",
    "影像检查": "CT",
    "药物": "Drug",
    "手术": "Sur"
}
type_str = { str_type[key]: key for key in str_type.keys()}

def C_trans_to_E(string):
    E_pun = u',.!?[]()<>"\'"\':;'
    C_pun = u'，。！？【】（）《》“‘”’：；'
    table= {ord(f): ord(t) for f, t in zip(C_pun, E_pun)}
    string = string.translate(table)
    return re.sub("[ |\r|\n|\\\]", "_", string)

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

def train_and_dev(X_train,y_train, X_dev, X_test,
                  model_name_or_path="bert-base-chinese", is_fit=True,
                  name="clinical"):

    if os.path.exists("tmp_y_{}_{}.pkl".format("dev", name)):
        return

    bertology_classifer = BertologyTokenClassifier(model_name_or_path=model_name_or_path,
                                                   do_cv=True, learning_rate=5e-5,
                                                   do_lower_case=True,
                                                   max_epochs=100, max_seq_length=128,
                                                   per_train_batch_size=8, per_val_batch_size=8,
                                                   classifier_type="LSTM_CRF", classifier_dropout=0.1,
                                                   schedule_type="cosine", weight_decay=0.1, warmup=0.01,
                                                   k_fold=5, patience=7, n_saved=5,
                                                   output_dir="results/{}".format(name))

    if is_fit:
        bertology_classifer.fit(X_train, y_train)

    def get_y(X, type):
        y = bertology_classifer.predict(X)
        assert len(X) == len(y)
        assert len(X) == len(y)
        print(len(X))
        print(y[0])
        pd.to_pickle(y, "tmp_y_{}_{}.pkl".format(type, name))

    get_y(X_dev, "dev")
    get_y(X_test, "test")

def alignment_X_y(X, y, cut_texts):
    y_align = []

    for i, X_ in enumerate(X):
        cut_index = cut_texts[i]
        if isinstance(cut_index, int):
            y_ = y[cut_index]
        else:
            y_ = []
            for index in cut_index:
                y_.extend(y[index])

        assert len(X_) == len(y_), 'i:{};text_len:{};while label_len:{}'.format(i, len(X_), len(y_))
        y_align.append(y_)

    assert len(X) == len(y_align)
    return y_align

def get_dev_bio_result(out_file, texts, originalTexts, cut_texts, type_vocab, name="clinical", type="dev"):
    y_pred = pd.read_pickle("tmp_y_{}_{}.pkl".format(type, name))

    X_align, y_align = texts, alignment_X_y(texts, y_pred, cut_texts)

    entity_data = []

    body_list = ['昏', '痛', '悸', '累', '、', '慌', '闷', '胀', ')', '泻', '晕', '纳', '痒', '干',
                 '水', '花', '增', '渴', ',', '，', '：', ':', '治', ';', '；', '占', '停',
                 '多', '硬', '"', '”', '等', '按', '迷', '中', '及', '内', '鸣', ')', '）',
                 '明', '.']

    test_list = ['_', '：', ":"]

    for i, (X_, y_) in enumerate(zip(X_align, y_align)):
        assert len(X_) == len(y_), 'i:{};text_len:{};while label_len:{}'.format(i, len(X_), len(y_))
        entities = []

        for k, label in enumerate(y_):
            if "-" in label:
                tag_1 = label.split("-")[0]
                tag_2 = label.split("-")[1]
                if tag_1 == "B":
                    start_pos = k
                    end_pos = k + 1
                    for j in range(start_pos+1, len(y_)):
                        if y_[j] == "I-" + tag_2:
                            end_pos += 1
                        else:
                            break

                    ## 规则
                    if name == "clinical":
                        entity = X_[start_pos: end_pos]
                        if len(entity) == 2 and tag_2 == "Body":
                            if entity[-1] in body_list:
                                entity = entity[0]
                                end_pos -= 1

                        if entity[-1] in test_list and tag_2 == "Test":
                            entity = entity[0]
                            end_pos -= 1

                    ## 词表相似度匹配
                    try:
                        start_pos, end_pos = vocab_simility(X_, start_pos, end_pos, tag_2, name, type, type_vocab)
                    except TypeError:
                        print("error: \n {}".format(X_))

                    entity = X_[start_pos: end_pos]
                    if name == "clinical":
                        label_type = type_str[tag_2]
                    else:
                        label_type = tag_2

                    tempObj = {
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                        "label_type": label_type
                    }
                    entities.append(tempObj)
                    entity_data.append((i + 1, entity, label_type, start_pos, end_pos))

                    msg = "text_id:{}\tentity:{}\tlabel_type:{}\tstart_pos:{}\tend_pos:{}". \
                        format(i + 1, entity, label_type, start_pos, end_pos)
                    print(msg)

        with open(out_file, 'a', encoding='utf-8') as f:
            s = json.dumps({
                "originalText": originalTexts[i],
                "entities": entities
            }, ensure_ascii=False)
            f.write(s)
            f.write("\n")

    tempDF = pd.DataFrame(data=entity_data, columns=['text_id', 'entity', 'label_type', 'start_pos', 'end_pos'])
    tempDF.to_csv("tmp_entities_{}_{}.csv".format(type, name), index=None)

def vocab_simility(text, start_pos, end_pos, tag, name, type, type_vocab):
    df_vocab = type_vocab[tag]
    entity = text[start_pos:end_pos]

    df_vocab['score'] = df_vocab['entity'].apply(lambda x: score_fn(x, entity))
    df_vocab.sort_values(by="score", ascending=False, inplace=True)
    top_5 = df_vocab['entity'].values[:5]

    if entity in top_5:
        return start_pos, end_pos
    else:
        pred_list = set(range(start_pos, end_pos))
        for vocab_entity in top_5:
            vocab_entity = C_trans_to_E(strQ2B(vocab_entity))
            if vocab_entity in text:
                try:
                    for match in re.finditer(vocab_entity, text):
                        start_pos_ = match.span()[0]
                        end_pos_ = start_pos_ + len(vocab_entity)
                        entity_list = set(range(start_pos_, end_pos_))
                        inter = pred_list.intersection(entity_list)
                        if len(inter) > 0 and abs(len(vocab_entity) - len(entity)) < 3:
                            msg = "vocab_entity:{} \t entity:{}\t tag:\t{}".format(vocab_entity, entity, tag)
                            print("vocab worked!\t{}".format(msg) )
                            with open("tmp_vocab_worked_{}_{}.csv".format(type, name), "a", encoding="utf-8") as f:
                                f.write("{},{},{}".format(vocab_entity, entity, tag))
                                f.write("\n")
                            return start_pos_, end_pos_
                        else:
                            start_pos, end_pos
                except re.error:
                    start_pos_ = text.index(vocab_entity)
                    end_pos_ = start_pos_ + len(vocab_entity)
                    entity_list = set(range(start_pos_, end_pos_))
                    inter = pred_list.intersection(entity_list)
                    if len(inter) > 0 and abs(len(vocab_entity) - len(entity)) < 3:
                        msg = "vocab_entity:{} \t entity:{}\t tag:\t{}".format(vocab_entity, entity, tag)
                        print("vocab worked!\t{}".format(msg))
                        with open("tmp_vocab_worked_{}_{}.csv".format(type, name), "a", encoding="utf-8") as f:
                            f.write("{},{},{}".format(vocab_entity, entity, tag))
                            f.write("\n")
                        return start_pos_, end_pos_
                    else:
                        return start_pos, end_pos
            else:
                return start_pos, end_pos

def score_fn(a,b):
    seq_match = SequenceMatcher(a=a, b=b)
    return round(seq_match.ratio(), ndigits=4)

def run_ner(name="clinical"):

    X_train, y_train, _ = pd.read_pickle("data/{}_ner_train.pkl".format(name))
    X_dev, cut_texts_dev, originalTexts_dev, texts_dev = pd.read_pickle("data/{}_ner_dev.pkl".format(name))
    X_test, cut_texts_test, originalTexts_test, texts_test = pd.read_pickle("data/{}_ner_test.pkl".format(name))

    root = Path("data")
    type_vocab = {}
    for file in root.glob("{}_*_vocab.csv".format(name)):
        vocab = pd.read_csv(file)
        tag = str(file).replace("data/{}_".format(name), "")
        tag = tag.replace("_vocab.csv", "")
        print(tag)
        type_vocab[tag] = vocab


    train_and_dev(X_train, y_train, X_dev, X_test,
                  model_name_or_path="/home/yfh/bertology_models/bert-base-chinese",
                  name=name, is_fit=True)

    get_dev_bio_result("{}_ner_dev_pred.txt".format(name), texts_dev, originalTexts_dev,
                       cut_texts_dev, type_vocab, name=name, type="dev")
    get_dev_bio_result("{}_ner_test_pred.txt".format(name), texts_test, originalTexts_test,
                       cut_texts_test, type_vocab, name=name, type="test")

if __name__ == '__main__':
    run_ner("clinical")
    run_ner("diabetes")