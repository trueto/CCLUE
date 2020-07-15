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

import pandas as pd

from bertology_sklearn import BertologyClassifier



def run_classifier(name="health_match", model_name_or_path="bert-base-chinese",
                   is_fit=True, max_len=64):

    train_df = pd.read_csv("data/{}_train.csv".format(name))
    dev_df = pd.read_csv("data/{}_dev.csv".format(name))
    test_df = pd.read_csv("data/{}_test.csv".format(name))

    if name == "health_match":
        X_train, y_train = train_df[["question1", "question2"]], train_df['label']
        X_dev, y_dev = dev_df[["question1", "question2"]], dev_df['label']
        X_test, y_test = test_df[["question1", "question2"]], test_df['label']

    else:
        X_train, y_train = train_df['text'], train_df['label']
        X_dev, y_dev = dev_df['text'], dev_df['label']
        X_test, y_test = test_df['text'], test_df['label']

    bert_cls = BertologyClassifier(model_name_or_path=model_name_or_path,
                                   do_lower_case=True, max_seq_length=max_len,
                                   output_dir="results/{}".format(name),
                                   per_train_batch_size=16, per_val_batch_size=16,
                                   do_cv=True, max_epochs=100, patience=7,
                                   n_saved=5, classifier_dropout=0.1,
                                   learning_rate=5e-5)

    if is_fit:
        bert_cls.fit(X_train, y_train)

    dev_socre = bert_cls.score(X_dev, y_dev)
    test_score = bert_cls.score(X_test, y_test)

    with open("{}_benchmark_score.txt".format(name), 'w', encoding="utf8") as w:
        w.write("{}\t验证集分数：\t{}".format(name, dev_socre))
        w.write("\n")
        w.write("{}\t测试集分数：\t{}".format(name, test_score))

if __name__ == '__main__':
    run_classifier("trail_classify", model_name_or_path="/home/yfh/bertology_models/bert-base-chinese",
                   max_len=64)
    run_classifier("health_match", model_name_or_path="/home/yfh/bertology_models/bert-base-chinese",
                   max_len=64)

    # train_df = pd.read_csv("data/{}_train.csv".format("trail_classify"))
    # print(train_df['text'].str.len().describe())
