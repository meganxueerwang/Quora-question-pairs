
# coding: utf-8

from __future__ import division
import pandas as pd
import numpy as np
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

plt.rc('figure',figsize = (12,6))
plt.rc('font', size = 18)
get_ipython().magic('matplotlib inline')

os.chdir('/Users/apple/Desktop/Programming/BIA/2017_Spring/BIA656/project/quora')

data = pd.read_csv("../quora/train.csv", nrows=100000)
data.head()

duplicated = data[data.is_duplicate == 1]
duplicated.reset_index(drop=True, inplace=True)
#reset_index
#remove some rows. As a result, I get a data frame in which index is something like that: 
#[1,5,6,10,11] and I would like to reset it to [0,1,2,3,4]
duplicated.head()

import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))

def preprocessing(string_data):              # pre-processing string
    """pre-processing, return word set reduce stopwords
    """
    string_data = re.sub('[\,\?,\(\)\.\:]', " ", string_data)
    string_data = re.sub(' +', " ", string_data)
    terms = set(string_data.lower().split(" ")) - stopwords - set(["","i'm"])
    return terms

q1 = data.question1.fillna("").apply(lambda x: preprocessing(x))
q2 = data.question2.fillna("").apply(lambda x: preprocessing(x))


# # using simple shared words and sentence length

share_words_num = []
diff_words_num = []
for i,j in zip(q1,q2):
    shared = len(i&j)
    diff = len(i|j) - shared
    share_words_num.append(shared)
    diff_words_num.append(diff)

data['q1_len'] = q1.apply(len)
data['q2_len'] = q2.apply(len)

data['shared'] = share_words_num
data['diff'] = diff_words_num


# # modify by extracting keywords using tf-idf¶

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def extract_tfidf_vector(string_list, keywords_num = 10):
    result = []
    vectorizer = CountVectorizer(min_df=2, stop_words='english')    
    q = vectorizer.fit_transform(string_list)
    features = vectorizer.get_feature_names()
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(q).toarray()
    for line in tfidf:
        tmp = np.argsort(line)[-keywords_num:]
        tmp_features = [features[c] for c in tmp]
        tfidf_features = [line[c] for c in tmp]
        result.append([tmp_features, tfidf_features])
    return result

get_ipython().magic('time q1_tfidf = extract_tfidf_vector(data.question1)')

q2_tfidf = extract_tfidf_vector(data.question2)

from scipy.spatial.distance import cosine, euclidean

tfidf_cosine = []
tfidf_euclidean = []
for i,j in zip(q1_tfidf,q2_tfidf):
    tfidf_cosine.append(cosine(i[1],j[1]))
    tfidf_euclidean.append(euclidean(i[1],j[1]))

data['tfidf_cosine'] = tfidf_cosine
data['tfidf_euclidean'] = tfidf_euclidean

from __future__ import division

# # build Jaccard similarity coefficient

def simi(lst1,lst2,order=False):
    if order == True:
        return sum([i==j for i,j in zip(lst1,lst2)])
    elif order == False:
        if set(lst1)|set(lst2):
            return len(set(lst1)&set(lst2))/len(set(lst1)|set(lst2))
        else:
            return 0.

data['keywords_simi_order'] = [simi(i[0],j[0],True) for i,j in zip(q1_tfidf,q2_tfidf)]
data['keywords_simi_disorder'] = [simi(i[0],j[0]) for i,j in zip(q1_tfidf,q2_tfidf)]

# back up 
stored_lables = ['id','is_duplicate','q1_len','q2_len','shared','diff',
                 'tfidf_cosine','tfidf_euclidean','keywords_simi_order','keywords_simi_disorder']
data[stored_lables].to_csv("../quora/features_for_training.csv")

q1_tfidf = np.stack(q1_tfidf)
q2_tfidf = np.stack(q2_tfidf)
np.save('../quora/q1_tfidf.npy',q1_tfidf)
np.save('../quora/q2_tfidf.npy',q2_tfidf)

# # add word vectors

from gensim.models import KeyedVectors
import numpy as np

wv = KeyedVectors.load_word2vec_format( "../quora/GoogleNews-vectors-negative300.bin", binary=True) 

def build_sentence_vector(wv, word_list, tfidf):
    sv = []
    for c in word_list:
        try:
            sv.append(wv[c])
        except:
            sv.append(np.zeros(300))
#     sv = [wv[c] for c in word_list]
    tmp = []
    for i,j in zip(sv,tfidf):
        tmp.append(i*float(j))
    return np.mean(tmp,axis=0)

def caculate_sentence_similarity(sv1,sv2,simitype='cosine'):
    if simitype == 'cosine':
        return cosine(sv1,sv2)
    elif simitype == 'euclidean':
        return euclidean(sv1,sv2)

def processing_senetence(q1_tfidf,q2_tfidf):
    sv_simi_cosine = []
    sv_simi_euclidean = []
    for a,b in zip(q1_tfidf,q2_tfidf):
        q1_sv = build_sentence_vector(wv,a[0],a[1])
        q2_sv = build_sentence_vector(wv,b[0],b[1])    
        sv_simi_cosine.append(caculate_sentence_similarity(q1_sv,q2_sv))
        sv_simi_euclidean.append(caculate_sentence_similarity(q1_sv,q2_sv,simitype='euclidean'))        
    return sv_simi_cosine,sv_simi_euclidean

sv_simi_cosine,sv_simi_euclidean = processing_senetence(q1_tfidf,q2_tfidf)

data['sv_simi_cosine'] = sv_simi_cosine
data['sv_simi_euclidean'] = sv_simi_euclidean

# back up 
stored_lables = ['id','is_duplicate','q1_len','q2_len','shared','diff','tfidf_cosine','tfidf_euclidean',
            'keywords_simi_order','keywords_simi_disorder','sv_simi_cosine','sv_simi_euclidean']
data[stored_lables].to_csv("../quora/features_for_training_add_sv.csv")

data.head()

features = ['q1_len','q2_len','shared','diff','tfidf_cosine','tfidf_euclidean',
            'keywords_simi_order','keywords_simi_disorder','sv_simi_cosine','sv_simi_euclidean']
target = ['is_duplicate']

g = data[features + target].groupby('is_duplicate')
g.apply(np.mean)
data.fillna(0,inplace=True)


# # Logisitic Regression Model

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
x_train, x_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.33, random_state=42)

regs = {
        'SVM': SVR(),
        'RF' : RandomForestRegressor(random_state=2017),
        'GB' : GradientBoostingRegressor(learning_rate=0.001,alpha=0.001)}

y_pred = {}
for name, reg in regs.items():
    reg.fit(x_train,y_train)
    y_pred[name] = reg.predict(x_test)
y_pred

loglosses = {}
for name, preds in y_pred.items():
    loglosses[name] = log_loss(y_test,preds)
print('loglosses:',loglosses)

from sklearn.linear_model import LogisticRegression
logR = LogisticRegression()
print(logR.fit(x_train, y_train))
logR_predict = logR.predict_proba(x_test)
LogR_pred = logR.predict(x_test)
print (logR_predict)

loglosses = log_loss(y_test,logR_predict)
print('logloss_logR:',loglosses)
y_pred_SVM = np.mean(y_pred['SVM'],axis=0)
y_pred_SVM

accuracy_LogR = [accuracy_score(y_test, LogR_pred)]
print('accuracy_LogR', accuracy_LogR)

accuracy_SVM = [accuracy_score(y_test,y_pred['SVM']>c) for c in np.arange(0,1,0.01)]
accuracy_SVM = np.mean(accuracy_SVM)
print('accuracy_SVM',accuracy_SVM)

accuracy_RF = [accuracy_score(y_test,y_pred['RF']>c) for c in np.arange(0,1,0.01)]
accuracy_RF = np.mean(accuracy_RF)
print('accuracy_RF',accuracy_RF)

accuracy_GB = [accuracy_score(y_test,y_pred['GB']>c) for c in np.arange(0,1,0.01)]
accuracy_GB = np.mean(accuracy_GB)
print('accuracy_GB',accuracy_GB)

#ROC
print(metrics.roc_auc_score(y_test, LogR_pred))
ROC_SVM = np.mean([metrics.roc_auc_score(y_test, y_pred['SVM']>c) for c in np.arange(0,1,0.01)])
print(ROC_SVM)
ROC_RF = np.mean([metrics.roc_auc_score(y_test, y_pred['RF']>c) for c in np.arange(0,1,0.01)])
print(ROC_RF)
ROC_GB = np.mean([metrics.roc_auc_score(y_test, y_pred['GB']>c) for c in np.arange(0,1,0.01)])
print(ROC_GB)

from matplotlib import colors as mcolors
fpr_logR, tpr_logR, _ = metrics.roc_curve(y_test,logR.predict_proba(x_test)[:,1])
precision_logR, recall_logR, thresholds_logR = metrics.precision_recall_curve(y_test,logR.predict_proba(x_test)[:,1])

fpr_SVM, tpr_SVM, _ = metrics.roc_curve(y_test,regs['SVM'].predict(x_test))
precision_SVM, recall_SVM, thresholds_SVM = metrics.precision_recall_curve(y_test,regs['SVM'].predict(x_test))

fpr_RF, tpr_RF, _ = metrics.roc_curve(y_test,regs['RF'].predict(x_test))
precision_RF, recall_RF, thresholds_RF = metrics.precision_recall_curve(y_test,regs['RF'].predict(x_test))

fpr_GB, tpr_GB, _ = metrics.roc_curve(y_test,regs['GB'].predict(x_test))
precision_GB, recall_GB, thresholds_GB = metrics.precision_recall_curve(y_test,regs['GB'].predict(x_test))

plt.plot(fpr_logR, tpr_logR, label='ROC Curve of LR', color = 'deepskyblue')
plt.plot(fpr_SVM, tpr_SVM, label='ROC Curve of SVM', color = 'dimgrey')
plt.plot(fpr_RF, tpr_RF, label='ROC Curve of RF', color = 'darkorange')
plt.plot(fpr_GB, tpr_GB, label='ROC Curve of GB', color = 'orchid')
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

plt.savefig("ROC.png", dpi = 600, bbox_inches='tight')

plt.clf()
plt.plot(recall_logR, precision_logR, label='Precision-Recall curve of LR', color = 'navy')
plt.plot(precision_SVM, recall_SVM, label='Precision-Recall curve of SVM', color = 'turquoise' )
plt.plot(precision_RF, recall_RF, label='Precision-Recall curve of RF', color = 'darkorange')
plt.plot(precision_GB, recall_GB, label='Precision-Recall curve of GB', color = 'cornflowerblue')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall')
plt.legend(loc="lower left")


plt.savefig("PR.png", dpi = 600, bbox_inches='tight')
plt.show()

plt.clf()
plt.plot(recall_logR, precision_logR, precision_SVM, recall_SVM, label='Precision-Recall curve of LR')
precision_logR, recall_logR, thresholds_logR = metrics.precision_recall_curve(test['class'],logR.predict_proba(test.ix[:,0:9])[:,1])

plt.plot(fpr_logR, tpr_logR, label='ROC Curve of Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic(LR)')
plt.legend(loc="lower right")
plt.show()

plt.clf()
plt.plot(recall_logR, precision_logR, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall of Logistic Regression(LR)')
plt.show()
# print(metrics.average_precision_score(test['class'], logR_predict))

y_pred['SVM']

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(data[features])

X = scaler.transform(data[features])
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Run cross-validation with a few hyper parameters
clf = LogisticRegression()
grid = {
    'C': [1e0]
}
cv = GridSearchCV(clf, grid, scoring='log_loss', n_jobs=-1, verbose=1)
cv.fit(x_train, y_train)

print (accuracy_score(y_test, y_pred))
print (log_loss(y_test, y_pred))

#Print validation results. 
#Here we see that the strongly regularized model has much worse negative log loss than the other two models, regardless of which regularizer we've used.

for i in range(1, len(cv.cv_results_['params'])+1):
    rank = cv.cv_results_['rank_test_score'][i-1]
    s = cv.cv_results_['mean_test_score'][i-1]
    sd = cv.cv_results_['std_test_score'][i-1]
    params = cv.cv_results_['params'][i-1]
    print("{0}. Mean validation neg log loss: {1:.3f} (std: {2:.3f}) - {3}".format(
        rank,
        s,
        sd,
        params
    ))

print(cv.best_params_)
print(cv.best_estimator_.coef_)

# ROC
# three different classifiers: a strongly regularized one and two with weaker regularization. 
# The heavily regularized model has parameters very close to zero and is actually worse than if we would pick the labels for our holdout samples randomly.

colors = ['y', 'g', 'b', 'y', 'k', 'c', 'm', 'brown', 'r']
lw = 1
Cs = [1e-6, 1e-4, 1e0]

plt.figure(figsize=(12,8))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for different classifiers')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
labels = []
for idx, C in enumerate(Cs):
    clf = LogisticRegression(C = C)
    clf.fit(X_train, y_train)
    print("C: {}, parameters {} and intercept {}".format(C, clf.coef_, clf.intercept_))
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=colors[idx])
    labels.append("C: {}, AUC = {}".format(C, np.round(roc_auc, 4)))

plt.legend(['random AUC = 0.5'] + labels)

# Precision & Recall

# Also used very commonly, but more often in cases where we have class-imbalance. We can see here, that there are a few positive samples that we can identify quite reliably. On in the medium and high recall regions we see that there are also positives samples that are harder to separate from the negatives.
pr, re, _ = precision_recall_curve(y_test, cv.best_estimator_.predict_proba(X_test)[:,1])
plt.figure(figsize=(12,8))
plt.plot(re, pr)
plt.title('PR Curve (AUC {})'.format(auc(re, pr)))
plt.xlabel('Recall')
plt.ylabel('Precision')
