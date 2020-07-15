# CCLUE
中文临床自然语言处理算法评估基准

A benchmark for Chinese ClinicalLanguage Understanding Evalution（CCLUE）

## CCLUE数据集


| 名称 | 场景 | 训练集 | 验证集 | 测试集| 评价指标 |
| : ------: | : ------: | : ------: | : ------: | : ------: | :------: |
| Clinical-NER | 电子病历 | 1871 | 267 | 536| 严格F1 |
| Diabetes-NER | 糖尿病指南或文献 | 6689 |955 | 1912|严格F1|
| Health-Match| 健康问答| 14000|2000|4000|Averaged F1|
| Trial-Classify| 临床试验| 26838| 3834| 7669| Averaged F1|


## BERT模型在CCLUE中的表现


| 名称 | 验证集F1 | 测试集F1 | 算法 |
| : ------: |: ------: | : ------: | :------: |
| Clinical-NER| 0.5715 | 0.5873 | BERT+LSTM+CRF多模融合|
| Diabetes-NER| 0.7861 | 0.7789	| BERT+LSTM+CRF多模融合 |
| Health-Match| 0.8864 | 0.8899 | BERT多模融合|
|Trial-Classify | 0.7869 | 0.8268| BERT多模融合|


## 脚本说明
脚本`ner_input_data.py`将数据准备为bert模型所需的文件格式

脚本`ner_train_dev_test.py`用于训练bert模型及获取模型对验证集和测试集的预测结果

脚本`diabetes_ner_metric.py`、`clinical_ner_metric.py`用于评估模型结果

脚本`classify_train_dev_test.py`用于训练bert模型及获取模型对验证集和测试集的预测结果

## 软件依赖
[bertology_sklearn](https://github.com/trueto/bertology_sklearn)
