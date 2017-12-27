
# coding: utf-8

# # load data

# In[2]:

import pandas as pd
import numpy as np
import re


# In[ ]:

# test in 100T rows


# In[3]:

data = pd.read_csv("../data/train.csv", nrows=100000)


# In[4]:

data.head()


# In[5]:

len(data)


# In[6]:

data.question1[0]


# In[7]:

data.question2[0]


# In[ ]:

# show duplicated data 


# In[18]:

duplicated = data[data.is_duplicate == 1]
duplicated.reset_index(drop=True, inplace=True)
duplicated.head()


# In[6]:

from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))


# In[7]:

def preprocessing(string_data):              # pre-processing string
    """pre-processing, return word set reduce stopwords
    """
    string_data = re.sub('[\,\?,\(\)\.\:]', " ", string_data)
    string_data = re.sub(' +', " ", string_data)
    terms = set(string_data.lower().split(" ")) - stopwords - set(["","i'm"])
    return terms


# In[8]:

q1 = data.question1.fillna("").apply(lambda x: preprocessing(x))
q2 = data.question2.fillna("").apply(lambda x: preprocessing(x))


# ## using simple shared words and sentence length

# In[9]:

share_words_num = []
diff_words_num = []
for i,j in zip(q1,q2):
    shared = len(i&j)
    diff = len(i|j) - shared
    share_words_num.append(shared)
    diff_words_num.append(diff)


# In[10]:

data['q1_len'] = q1.apply(len)
data['q2_len'] = q2.apply(len)

data['shared'] = share_words_num
data['diff'] = diff_words_num


# # modify by extracting keywords using tf-idf

# In[11]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[15]:

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


# In[16]:

get_ipython().magic('time q1_tfidf = extract_tfidf_vector(data.question1)')


# In[20]:

q2_tfidf = extract_tfidf_vector(data.question2)


# In[25]:

from scipy.spatial.distance import cosine, euclidean


# In[27]:

tfidf_cosine = []
tfidf_euclidean = []
for i,j in zip(q1_tfidf,q2_tfidf):
    tfidf_cosine.append(cosine(i[1],j[1]))
    tfidf_euclidean.append(euclidean(i[1],j[1]))


# In[28]:

data['tfidf_cosine'] = tfidf_cosine
data['tfidf_euclidean'] = tfidf_euclidean


# In[177]:

from __future__ import division


# In[285]:

# build Jaccard similarity coefficient


# In[44]:

def simi(lst1,lst2,order=False):
    if order == True:
        return sum([i==j for i,j in zip(lst1,lst2)])
    elif order == False:
        if set(lst1)|set(lst2):
            return len(set(lst1)&set(lst2))/len(set(lst1)|set(lst2))
        else:
            return 0.


# In[47]:

data['keywords_simi_order'] = [simi(i[0],j[0],True) for i,j in zip(q1_tfidf,q2_tfidf)]
data['keywords_simi_disorder'] = [simi(i[0],j[0]) for i,j in zip(q1_tfidf,q2_tfidf)]


# In[98]:

# back up 
stored_lables = ['id','is_duplicate','q1_len','q2_len','shared','diff',
                 'tfidf_cosine','tfidf_euclidean','keywords_simi_order','keywords_simi_disorder']
data[stored_lables].to_csv("../data/features_for_training.csv")


# In[101]:

q1_tfidf = np.stack(q1_tfidf)
q2_tfidf = np.stack(q2_tfidf)
np.save('../data/q1_tfidf.npy',q1_tfidf)
np.save('../data/q2_tfidf.npy',q2_tfidf)


# ## add word vectors

# In[103]:

from gensim.models import KeyedVectors
import numpy as np

wv = KeyedVectors.load_word2vec_format( "/Users/Nico/working_project_nico/GoogleNews-vectors-negative300.bin.gz", binary=True) 


# In[135]:

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


# In[124]:

def caculate_sentence_similarity(sv1,sv2,simitype='cosine'):
    if simitype == 'cosine':
        return cosine(sv1,sv2)
    elif simitype == 'euclidean':
        return euclidean(sv1,sv2)


# In[125]:

def processing_senetence(q1_tfidf,q2_tfidf):
    sv_simi_cosine = []
    sv_simi_euclidean = []
    for a,b in zip(q1_tfidf,q2_tfidf):
        q1_sv = build_sentence_vector(wv,a[0],a[1])
        q2_sv = build_sentence_vector(wv,b[0],b[1])    
        sv_simi_cosine.append(caculate_sentence_similarity(q1_sv,q2_sv))
        sv_simi_euclidean.append(caculate_sentence_similarity(q1_sv,q2_sv,simitype='euclidean'))        
    return sv_simi_cosine,sv_simi_euclidean


# In[137]:

sv_simi_cosine,sv_simi_euclidean = processing_senetence(q1_tfidf,q2_tfidf)


# In[140]:

data['sv_simi_cosine'] = sv_simi_cosine
data['sv_simi_euclidean'] = sv_simi_euclidean


# In[229]:

# back up 
stored_lables = ['id','is_duplicate','q1_len','q2_len','shared','diff','tfidf_cosine','tfidf_euclidean',
            'keywords_simi_order','keywords_simi_disorder','sv_simi_cosine','sv_simi_euclidean']
data[stored_lables].to_csv("../data/features_for_training_add_sv.csv")


# ## take a look at the data 

# In[230]:

data.head()


# In[146]:

g = data[features + target].groupby('is_duplicate')


# In[147]:

g.apply(np.mean)


# In[149]:

data.fillna(0,inplace=True)


# ## build models

# In[238]:

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, log_loss


# In[239]:

features = ['q1_len','q2_len','shared','diff','tfidf_cosine','tfidf_euclidean',
            'keywords_simi_order','keywords_simi_disorder','sv_simi_cosine','sv_simi_euclidean']
target = ['is_duplicate']


# In[240]:

x_train, x_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.33, random_state=42)


# In[241]:

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[242]:

regs = {'reg1': SVR(),
        'reg2' : RandomForestRegressor(random_state=2017),
        'reg3' : GradientBoostingRegressor(learning_rate=0.001,alpha=0.001)}


# In[ ]:

# could tune model individually


# In[243]:

y_pred = {}
for name, reg in regs.items():
    reg.fit(x_train,y_train)
    y_pred[name] = reg.predict(x_test)


# In[244]:

y_pred


# In[266]:

loglosses = {}
for name, preds in y_pred.items():
    loglosses[name] = log_loss(y_test,preds)


# In[267]:

loglosses


# In[270]:

y_pred_ensambled = np.mean(y_pred.values(),axis=0)  # could add weights 


# In[272]:

log_loss(y_test, y_pred_ensambled)


# In[273]:

accuracy = [accuracy_score(y_test,y_pred_ensambled>c) for c in np.arange(0,1,0.01)]


# In[65]:

import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')


# In[274]:

plt.plot(accuracy)


# In[284]:

imporance = pd.DataFrame(regs['reg2'].feature_importances_, index=features).sort_values(0,ascending=True)
imporance.plot.barh()


# In[ ]:

## could remove the usee less features 

