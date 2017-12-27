#!/usr/local/bin/python
# -*- coding: utf-8 -*-


from __future__ import division
import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import cosine, euclidean
from gensim.models import KeyedVectors
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


__author__ = 'Nico Zheng'
__email__ = 'nico921113@gmail.com'


sns.set_style("white")

plt.rc('figure',figsize=(12,6))
plt.rc('font', size=18)


def preprocessing(string_data):              # pre-processing string
    string_data = re.sub('[\,\?,\(\)\.\:]', " ", string_data)
    string_data = re.sub(' +', " ", string_data)
    #     terms = set(string_data.lower().split(" ")) - stopwords - set(["","i'm"])
    terms = set(string_data.lower().split(" ")) - set(["","i'm"])
    return terms


def simi(lst1,lst2,order=False):  # Jacarrd Similarity
    if order is True:
        return sum([i == j for i,j in zip(lst1,lst2)])
    elif order is False:
        if set(lst1) | set(lst2):
            return len(set(lst1) & set(lst2))/len(set(lst1) | set(lst2))
        else:
            return 0.


def build_sentence_vector(wv, word_list):
    sv = []
    if word_list:
        for c in word_list:
            try:
                sv.append(wv[c])
            except:
                sv.append(np.zeros(300))
        return np.mean(sv,axis=0)
    else:
        return np.zeros(300)


def caculate_sentence_similarity(sv1,sv2,simitype='cosine'):
    if simitype == 'cosine':
        return cosine(sv1,sv2)
    elif simitype == 'euclidean':
        return euclidean(sv1,sv2)


def processing_senetence(q1,q2):
    q1 = list(q1)
    q2 = list(q2)
    sv_simi_cosine = []
    sv_simi_euclidean = []
#     count = 0
    for a,b in zip(q1,q2):
        q1_sv = build_sentence_vector(wv,a)
        q2_sv = build_sentence_vector(wv,b)
        sv_simi_cosine.append(caculate_sentence_similarity(q1_sv,q2_sv))
        sv_simi_euclidean.append(caculate_sentence_similarity(q1_sv,q2_sv,simitype='euclidean'))
#         count+=1
    return sv_simi_cosine,sv_simi_euclidean


def get_xgb_imp(xgb, feat_names):
    imp_vals = xgb.get_fscore()
    total = sum(imp_vals.values())
    return {k:v/total for k,v in imp_vals.items()}


def generate_features(s1,s2):
    feature = []
    s1,s2 = preprocessing(s1), preprocessing(s2)
    feature = feature+[len(s1),len(s2)]
    feature = feature+[len(i & j),len(i | j) - len(i & j)]
    s1_sv,s2_sv = build_sentence_vector(wv,s1), build_sentence_vector(wv,s2)
    feature = feature+[caculate_sentence_similarity(s1_sv,s2_sv), caculate_sentence_similarity(s1_sv,s2_sv,simitype='euclidean')]
    return pd.DataFrame(feature,index=bst.feature_names).T


if __name__ == '__main__':
    data = pd.read_csv("../data/train.csv", nrows=100000)
    q1 = data.question1.fillna("").apply(lambda x: preprocessing(x))
    q2 = data.question2.fillna("").apply(lambda x: preprocessing(x))
    # ## using simple shared words and sentence length
    share_words_num = []
    diff_words_num = []
    for i,j in zip(q1,q2):
        shared = len(i & j)
        diff = len(i | j) - shared
        share_words_num.append(shared)
        diff_words_num.append(diff)

    data['q1_len'] = q1.apply(len)
    data['q2_len'] = q2.apply(len)
    data['shared'] = share_words_num
    data['diff'] = diff_words_num

    # add word vectors
    wv = KeyedVectors.load_word2vec_format("/Users/Nico/working_project_nico/GoogleNews-vectors-negative300.bin.gz", binary=True)
    sv_simi_cosine, sv_simi_euclidean = processing_senetence(q1,q2)

    data['sv_simi_cosine'] = sv_simi_cosine
    data['sv_simi_euclidean'] = sv_simi_euclidean

    # back up
    stored_lables = ['id','is_duplicate','q1_len','q2_len','shared','diff','sv_simi_cosine','sv_simi_euclidean']
    data[stored_lables].to_csv("../data/features_for_training_add_sv_demo_use.csv")
    data.fillna(0,inplace=True)

    # ## build models
    features = ['q1_len','q2_len','shared','diff','sv_simi_cosine','sv_simi_euclidean']
    target = ['is_duplicate']

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.33, random_state=42)

    # Set our parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 4
    result = {}

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_test = xgb.DMatrix(x_test, label=y_test)

    watchlist = [(d_train, 'train'), (d_test, 'test')]

    bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10,callbacks=[xgb.callback.record_evaluation(result)])
    d_test = xgb.DMatrix(x_test)
    p_test = bst.predict(d_test)

    # plot train and test logloss
    result = pd.DataFrame([result['train']['logloss'], result['test']['logloss']], index=['train_logloss','test_logloss']).T
    result.plot.line()
    plt.title('Gradient Boosting Train and Test Logloss')
    plt.xlabel('epoch')
    plt.ylabel('logloss')
    plt.show()
    # plt.savefig("Gradient Boosting Train and Test Logloss.png", dpi = 600, bbox_inches='tight')

    # plot feature importance
    feature_importance = get_xgb_imp(bst, bst.feature_names)
    feature_importance = pd.DataFrame(feature_importance.values(), index=feature_importance.keys(), columns=['feature importance'])
    feature_importance.sort_values('feature importance').plot.barh()
    plt.title('Gradient Boosting Feature Importance')
    plt.show()
    # plt.savefig("Gradient Boosting Feature Importance.png", dpi = 600, bbox_inches='tight')


    # plot accuracy
    accuracy = [accuracy_score(y_test,p_test > c) for c in np.arange(0,1,0.01)]
    accuracy_steps = pd.DataFrame(accuracy, index=np.arange(0,1,0.01), columns=['Accuracy'])
    accuracy_steps.plot()
    plt.xlabel('predict value threshold')
    plt.ylabel('accuracy on testing set')
    plt.title("Gradient Boosting Accuracy")
    plt.show()
    # plt.savefig("Gradient Boosting Accuracy.png", dpi = 600, bbox_inches='tight')
    

    # save model
    bst.save_model("demo.model")
    print '\n\n\n'    

    need_furture_input = raw_input('Would you like to test additional questions? [y/n]  ')
    while need_furture_input == 'y':
        test1 = raw_input('Please input test sentence 1:  ')
        test2 = raw_input('please input test sentence 2:  ')
        print '________\n\n'
        print "Prediction on test sentence pair is %s" % bst.predict(xgb.DMatrix(generate_features(test1, test2)))[0]
        print '________\n\n'
        need_furture_input = raw_input('Would you like to test additional questions? [y/n]   ')

    # # further input
    # test1 = "Do I have a chance of getting a job as a machine learning engineer or developer or even a programmer?"
    # test2 = "What is the future of deep learning? Are most machine learning experts turning to deep learning?"
    #
    # test1 = "How to invest in China?"
    # test2 = "How to invest in India?"
    #
    # test1 = "How do I learn machine learning?"
    # test2 = "How to learn machine learning?"
    #
    # test1 = "How can I be a good Data Scientist?"
    # test2 = "What should I do to be a good data scientist?"
    #
    # bst.predict(xgb.DMatrix(generate_features(test1, test2)))[0]
