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

# coding: utf-8

import json
# In[1]:

import collections


# In[2]:

Result = collections.namedtuple("Result", ["score", "message"])


# In[3]:
en_list = ['Amount', 'Anatomy', 'Disease', 'Drug', 'Duration', 'Frequency', 'Level', 'Method', 'Operation',
           'Reason', 'SideEff', 'Symptom', 'Test', 'Test_Value', 'Treatment']

def ccks_metric(sub_file, result_file):
    # if sub_file[-3:] == 'zip':
    #    sub_file = extract_zip(sub_file, file_type='.txt')
    with open(sub_file, 'r', encoding="utf8") as f1, open(result_file, 'r', encoding="utf8") as f2:
        sub_data = f1.readlines()
        res_data = f2.readlines()
    dict_sub = {}
    dict_res = {}
    row = 0
    row_line = 0
    for sub_line in sub_data:
        row += 1
        if len(sub_line.strip()) > 0:
            row_line += 1
            temp_Obj = json.loads(sub_line, encoding="utf8")
            dict_sub[temp_Obj['originalText']] = temp_Obj['entities']

    for res_line in res_data:
        if len(res_line.strip()) > 0:
            temp_Obj = json.loads(res_line, encoding="utf8")
            dict_res[temp_Obj['originalText']] = temp_Obj['entities']

    # if row_line != len(dict_res):
    #    return Result(-1, 'out of data')

    en_dict = {en: {} for en in en_list}
    en_g = {en: 0 for en in en_list}
    overall_g = 0

    for row_id in dict_res:
        if row_id not in dict_sub:
            return Result(-1, 'Incorrect ID in line: ' + str(row_id))
        t_lst = dict_res[row_id]
        for item in t_lst:
            overall_g += 1

            label_type = item["label_type"]
            en_g[label_type] += 1
            if row_id not in en_dict[label_type]:
                en_dict[label_type][row_id] = []
                en_dict[label_type][row_id].append(item)
            else:
                en_dict[label_type][row_id].append(item)

    en_s, overall_s = {en:0 for en in en_list}, 0
    en_r, overall_r = {en:0 for en in en_list}, 0

    predict, en_body = 0, {en:0 for en in en_list}

    for row_id in dict_sub:
        if row_id not in dict_res:
            return Result(-1, ("unknown id:" + row_id))
        s_lst = dict_sub[row_id]
        predict += len(s_lst)
        for item in s_lst:
            label_type = item["label_type"]
            en_body[label_type] += 1

            if row_id not in en_dict[label_type]:
                continue

            if item in en_dict[label_type][row_id]:
                en_s[label_type] += 1
                overall_s += 1
                en_r[label_type] += 1
                overall_r += 1
                en_dict[label_type][row_id].remove(item)

            else:
                for gold in en_dict[label_type][row_id]:
                    if max(int(item["start_pos"]), int(gold["start_pos"])) <= min(int(item["end_pos"]),int(gold["end_pos"])):
                        en_dict[label_type][row_id].remove(gold)
                        en_r[label_type] += 1
                        overall_r += 1
                        break

    precision, recall, f1 = {}, {}, {},

    for label_type in en_list:
        if en_body[label_type] == 0:
            precision['{}_s'.format(label_type)] = 0
            precision['{}_r'.format(label_type)] = 0
        else:
            precision['{}_s'.format(label_type)] = en_s[label_type] / en_body[label_type]
            precision['{}_r'.format(label_type)] = en_r[label_type] / en_body[label_type]

    if predict == 0:
        precision['overall_s'] = 0
    else:
        precision['overall_s'] = overall_s / predict

    if predict == 0:
        precision['overall_r'] = 0
    else:
        precision['overall_r'] = overall_r / predict

    for label_type in en_list:
        recall["{}_s".format(label_type)] = en_s[label_type] / en_g[label_type]
        recall["{}_r".format(label_type)] = en_r[label_type] / en_g[label_type]

    recall['overall_s'] = overall_s / overall_g
    recall['overall_r'] = overall_r / overall_g

    for item in precision:
        f1[item] = 2 * precision[item] * recall[item] / (precision[item] + recall[item]) \
            if (precision[item] + recall[item]) != 0 else 0

    s = ""
    for label_type in en_list:
        s += "{}_s:\t{} {}_r:\t{}\n".format(label_type, [precision['{}_s'.format(label_type)], recall['{}_s'.format(label_type)],
                                             f1['{}_s'.format(label_type)]], label_type, [precision['{}_r'.format(label_type)],
                                                                                         recall['{}_r'.format(label_type)],
                                                                                          f1['{}_r'.format(label_type)]])

    return Result(f1['overall_s'], s)


# In[5]:

if __name__ == '__main__':
    #
    print(ccks_metric('pred/diabetes_ner_dev_pred.txt', 'data/diabetes_ner_dev.txt'))
    print(ccks_metric('pred/diabetes_ner_test_pred.txt', 'data/diabetes_ner_test.txt'))




# In[ ]:



