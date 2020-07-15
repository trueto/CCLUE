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

    if row_line != len(dict_res):
        return Result(-1, 'out of data')

    anatomy_dict, disease_dict, test_dict, ct_dict, drugs_dict, operation_dict = {}, {}, {}, {}, {}, {}
    anatomy_g, disease_g, test_g, ct_g, drugs_g, operation_g,  overall_g= 0, 0, 0, 0, 0, 0, 0

    for row_id in dict_res:
        if row_id not in dict_sub:
            return Result(-1, 'Incorrect ID in line: ' + str(row_id))
        t_lst = dict_res[row_id]
        for item in t_lst:
            overall_g += 1

            if item["label_type"] == '解剖部位':
                anatomy_g += 1
                if row_id not in anatomy_dict:
                    anatomy_dict[row_id] = []
                    anatomy_dict[row_id].append(item)
                else:
                    anatomy_dict[row_id].append(item)

            elif item["label_type"] == '疾病和诊断':
                disease_g += 1
                if row_id not in disease_dict:
                    disease_dict[row_id] = []
                    disease_dict[row_id].append(item)
                else:
                    disease_dict[row_id].append(item)

            elif item["label_type"] == '实验室检验':
                test_g += 1
                if row_id not in test_dict:
                    test_dict[row_id] = []
                    test_dict[row_id].append(item)
                else:
                    test_dict[row_id].append(item)

            elif item["label_type"] == '影像检查':
                ct_g += 1
                if row_id not in ct_dict:
                    ct_dict[row_id] = []
                    ct_dict[row_id].append(item)
                else:
                    ct_dict[row_id].append(item)

            elif item["label_type"] == '药物':
                drugs_g += 1
                if row_id not in drugs_dict:
                    drugs_dict[row_id] = []
                    drugs_dict[row_id].append(item)
                else:
                    drugs_dict[row_id].append(item)
            elif item["label_type"] == '手术':
                operation_g += 1
                if row_id not in operation_dict:
                    operation_dict[row_id] = []
                    operation_dict[row_id].append(item)
                else:
                    operation_dict[row_id].append(item)
            else:
                return Result(-1, ("unknown label: " + str(item)))

    anatomy_s, disease_s, test_s, ct_s, drugs_s, operation_s, overall_s = 0, 0, 0, 0, 0, 0, 0
    anatomy_r, disease_r, test_r, ct_r, drugs_r, operation_r, overall_r = 0, 0, 0, 0, 0, 0, 0
    predict, anatomy_body, disease_body, test_body, ct_body, drugs_body, operation_body = 0, 0, 0, 0, 0, 0, 0

    for row_id in dict_sub:
        if row_id not in dict_res:
            return Result(-1, ("unknown id:" + row_id))
        s_lst = dict_sub[row_id]
        predict += len(s_lst)
        for item in s_lst:
            if item["label_type"] == '解剖部位':
                anatomy_body += 1
                if row_id not in anatomy_dict:
                    continue
                if item in anatomy_dict[row_id]:
                    anatomy_s += 1
                    overall_s += 1
                    anatomy_r += 1
                    overall_r += 1
                    anatomy_dict[row_id].remove(item)
                else:
                    for gold in anatomy_dict[row_id]:
                        if max(int(item["start_pos"]), int(gold["start_pos"])) <= min(int(item["end_pos"]), int(gold["end_pos"])):
                            anatomy_dict[row_id].remove(gold)
                            anatomy_r += 1
                            overall_r += 1
                            break
            elif item["label_type"] == '疾病和诊断':
                disease_body += 1
                if row_id not in disease_dict:
                    continue
                if item in disease_dict[row_id]:
                    disease_s += 1
                    overall_s += 1
                    disease_r += 1
                    overall_r += 1
                    disease_dict[row_id].remove(item)
                else:
                    for gold in disease_dict[row_id]:
                        if max(int(item["start_pos"]), int(gold["start_pos"])) <= min(int(item["end_pos"]), int(gold["end_pos"])):
                            disease_dict[row_id].remove(gold)
                            disease_r += 1
                            overall_r += 1
                            break
            elif item["label_type"] == '实验室检验':
                test_body += 1
                if row_id not in test_dict:
                    continue
                if item in test_dict[row_id]:
                    test_s += 1
                    overall_s += 1
                    test_r += 1
                    overall_r += 1
                    test_dict[row_id].remove(item)
                else:
                    for gold in test_dict[row_id]:
                        if max(int(item["start_pos"]), int(gold["start_pos"])) <= min(int(item["end_pos"]), int(gold["end_pos"])):
                            test_dict[row_id].remove(gold)
                            test_r += 1
                            overall_r += 1
                            break
            elif item["label_type"] == '影像检查':
                ct_body += 1
                if row_id not in ct_dict:
                    continue
                if item in ct_dict[row_id]:
                    ct_s += 1
                    overall_s += 1
                    ct_r += 1
                    overall_r += 1
                    ct_dict[row_id].remove(item)
                else:
                    for gold in ct_dict[row_id]:
                        if max(int(item["start_pos"]), int(gold["start_pos"])) <= min(int(item["end_pos"]), int(gold["end_pos"])):
                            ct_dict[row_id].remove(gold)
                            ct_r += 1
                            overall_r += 1
                            break
            elif item["label_type"] == '药物':
                drugs_body += 1
                if row_id not in drugs_dict:
                    continue
                if item in drugs_dict[row_id]:
                    drugs_s += 1
                    overall_s += 1
                    drugs_r += 1
                    overall_r += 1
                    drugs_dict[row_id].remove(item)
                else:
                    for gold in drugs_dict[row_id]:
                        if max(int(item["start_pos"]), int(gold["start_pos"])) <= min(int(item["end_pos"]), int(gold["end_pos"])):
                            drugs_dict[row_id].remove(gold)
                            drugs_r += 1
                            overall_r += 1
                            break
            elif item["label_type"] == '手术':
                operation_body += 1
                if row_id not in operation_dict:
                    continue
                if item in operation_dict[row_id]:
                    operation_s += 1
                    overall_s += 1
                    operation_r += 1
                    overall_r += 1
                    operation_dict[row_id].remove(item)
                else:
                    for gold in operation_dict[row_id]:
                        if max(int(item["start_pos"]), int(gold["start_pos"])) <= min(int(item["end_pos"]), int(gold["end_pos"])):
                            operation_dict[row_id].remove(gold)
                            operation_r += 1
                            overall_r += 1
                            break
            else:
                return Result(-1, ("unknown label: " + str(item)))

    precision, recall, f1 = {}, {}, {},

    if anatomy_body == 0:
        precision['anatomy_s'] = 0
    else:
        precision['anatomy_s'] = anatomy_s / anatomy_body

    if disease_body == 0:
        precision['disease_s'] = 0
    else:
        precision['disease_s'] = disease_s / disease_body

    if test_body == 0:
        precision['test_s'] = 0
    else:
        precision['test_s'] = test_s / test_body

    if ct_body == 0:
        precision['ct_s'] = 0
    else:
        precision['ct_s'] = ct_s / ct_body

    if drugs_body == 0:
        precision['drugs_s'] = 0
    else:
        precision['drugs_s'] = drugs_s / drugs_body

    if operation_body == 0:
        precision['operation_s'] = 0
    else:
        precision['operation_s'] = operation_s / operation_body

    if predict == 0:
        precision['overall_s'] = 0
    else:
        precision['overall_s'] = overall_s / predict

    if anatomy_body == 0:
        precision['anatomy_r'] = 0
    else:
        precision['anatomy_r'] = anatomy_r / anatomy_body

    if disease_body == 0:
        precision['disease_r'] = 0
    else:
        precision['disease_r'] = disease_r / disease_body

    if test_body == 0:
        precision['test_r'] = 0
    else:
        precision['test_r'] = test_r / test_body

    if ct_body == 0:
        precision['ct_r'] = 0
    else:
        precision['ct_r'] = ct_r / ct_body

    if drugs_body == 0:
        precision['drugs_r'] = 0
    else:
        precision['drugs_r'] = drugs_r / drugs_body
    if operation_body == 0:
        precision['operation_r'] = 0
    else:
        precision['operation_r'] = operation_r / operation_body
    if predict == 0:
        precision['overall_r'] = 0
    else:
        precision['overall_r'] = overall_r / predict

    recall['anatomy_s'] = anatomy_s / anatomy_g
    recall['disease_s'] = disease_s / disease_g
    recall['test_s'] = test_s / test_g
    recall['ct_s'] = ct_s / ct_g
    recall['drugs_s'] = drugs_s / drugs_g
    recall['operation_s'] = operation_s / operation_g
    recall['overall_s'] = overall_s / overall_g

    recall['anatomy_r'] = anatomy_r / anatomy_g
    recall['disease_r'] = disease_r / disease_g
    recall['test_r'] = test_r / test_g
    recall['ct_r'] = ct_r / ct_g
    recall['drugs_r'] = drugs_r / drugs_g
    recall['operation_r'] = operation_r / operation_g
    recall['overall_r'] = overall_r / overall_g

    for item in precision:
        f1[item] = 2 * precision[item] * recall[item] / (precision[item] + recall[item]) \
            if (precision[item] + recall[item]) != 0 else 0

    anatomy_s = [precision['anatomy_s'], recall['anatomy_s'], f1['anatomy_s']]
    anatomy_r = [precision['anatomy_r'], recall['anatomy_r'], f1['anatomy_r']]
    disease_s = [precision['disease_s'], recall['disease_s'], f1['disease_s']]
    disease_r = [precision['disease_r'], recall['disease_r'], f1['disease_r']]
    test_s = [precision['test_s'], recall['test_s'], f1['test_s']]
    test_r = [precision['test_r'], recall['test_r'], f1['test_r']]
    ct_s = [precision['ct_s'], recall['ct_s'], f1['ct_s']]
    ct_r = [precision['ct_r'], recall['ct_r'], f1['ct_r']]
    drugs_s = [precision['drugs_s'], recall['drugs_s'], f1['drugs_s']]
    drugs_r = [precision['drugs_r'], recall['drugs_r'], f1['drugs_r']]
    operation_s = [precision['operation_s'], recall['operation_s'], f1['operation_s']]
    operation_r = [precision['operation_r'], recall['operation_r'], f1['operation_r']]
    overall_r = [precision['overall_r'], recall['overall_r'], f1['overall_r']]
    overall_s = [precision['overall_s'], recall['overall_s'], f1['overall_s']]

    s = 'anatomy_s: ' + str(anatomy_s) + ' anatomy_r: ' + str(anatomy_r) + ' disease_s: ' + str(disease_s) + \
        ' disease_r: ' + str(disease_r) + ' test_s: ' + str(test_s) + ' test_r: ' + str(test_r) + \
        ' ct_s: ' + str(ct_s) + ' ct_r: ' + str(ct_r) + ' drugs_s: ' + str(drugs_s) + ' drugs_r: ' + str(drugs_r) \
        + ' operation_s: ' + str(operation_s) + ' operation_r: ' + str(operation_r) + ' overall_r: ' \
        + str(overall_r) + ' overall_s: ' + str(overall_s)

    return Result(f1['overall_s'], s)


# In[5]:

if __name__ == '__main__':
    # 'Amount', 'Anatomy', 'Disease', 'Drug', 'Duration', 'Frequency', 'Level', 'Method', 'Operation', 'Reason', 'SideEff', 'Symptom', 'Test', 'Test_Value', 'Treatment'
    print(ccks_metric('pred/clinical_ner_dev_pred.txt', 'data/clinical_ner_dev.txt'))
    print(ccks_metric('pred/clinical_ner_test_pred.txt', 'data/clinical_ner_test.txt'))


# In[ ]:



