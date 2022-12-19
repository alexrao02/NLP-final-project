import time
import wikipedia as wiki
import json
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from gensim.parsing.preprocessing import preprocess_string
from gensim.models import Word2Vec
import gensim.downloader
'''
Document Retrieval
Given a question, the document retriever have to return the most likely k documents that contain the answer to the question.
'''

# how Wikipedia works
k = 5
question = "What are the tourist hotspots in Portugal?"
# results = wiki.search(question, results=k)
# print('Question:', question)
# print('Pages:  ', results)

# 数据预处理
# list the available data
for dirname, _, filenames in os.walk('kaggle\input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In [5]
# based on: https://www.kaggle.com/code/sanjay11100/squad-stanford-q-a-json-to-pandas-dataframe
def squad_json_to_dataframe(file_path, record_path=['data', 'paragraphs', 'qas', 'answers']):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    """
    file = json.loads(open(file_path).read())
    # parsing different level's in the json file
    # pd.json_normalize --→ 将半结构化数据json规范化为平面表
    js = pd.json_normalize(file, record_path)
    m = pd.json_normalize(file, record_path[:-1])
    r = pd.json_normalize(file, record_path[:-2])
    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    m['context'] = idx
    data = m[['id', 'question', 'context', 'answers']].set_index('id').reset_index()
    data['c_id'] = data['context'].factorize()[0]
    return data


# In [6]
# loading the data
file_path = 'D:/作业/NLP大作业/代码区/kaggle/input/stanford-question-answering-dataset/train-v1.1.json'
data = squad_json_to_dataframe(file_path)
print(data)

# In [7]
# how many documents do we have?
print(data['c_id'].unique().size)

# In [8]
# 去重
documents = data[['context', 'c_id']].drop_duplicates().reset_index(drop=True)
print(documents)

'''
Document Retrieval
构建vectorizer, documents、questions --→ vector
输入问题、转化为问题矩阵、比对相似度、返回k个相似度最高的文档矩阵
'''
# TF-IDF 反向文档频率
# In [9]
# defining the TF-IDF
tfidf_configs = {
    'lowercase': True,
    'analyzer': 'word',
    'stop_words': 'english',
    'binary': True,
    'max_df': 0.9,
    'max_features': 10_000
}
# defining the number of documents to retrieve
retriever_configs = {
    'n_neighbors': 3,
    'metric': 'cosine'
}

# defining our pipeline
embedding = TfidfVectorizer(**tfidf_configs)
retriever = NearestNeighbors(**retriever_configs)

# In [10]
# let's train the model to retrieve the document id 'c_id'
X = embedding.fit_transform(documents['context'])
print(retriever.fit(X, documents['c_id']))

# In [11]
# Let's test the vectorizer, what information our model is using to extract the vector?
def transform_text(vectorizer, text):
    '''
    Print the text and the vector[TF-IDF]
    vectorizer: sklearn.vectorizer
    text: str
    '''
    print('Text:', text)
    vector = vectorizer.transform([text])
    vector = vectorizer.inverse_transform(vector)
    print('Vect:', vector)


# In [12]
# vectorize the question
transform_text(embedding, question)

# In [13]
# What is the most similar document to this question?
# predict the most similar document
X = embedding.transform([question])
c_id = retriever.kneighbors(X, return_distance=False)[0][0]
selected = documents.iloc[c_id]['context']

# vectorize the document
transform_text(embedding, selected)

'''
dtype('<U1')
dtype:表示数组的数据类型
第一个字符是字节序，< 表示小端，> 表示大端，| 表示平台的字节序；U是上表中的最后一行Unicode的意思；1代表长度字符串的长度
'''


# In [14]
print('评估')
# predict one document for each question
tis1 = time.perf_counter()
X = embedding.transform(data['question'])
y_test = data['c_id']
y_pred = retriever.kneighbors(X, return_distance=False)
tis2 = time.perf_counter()
print(tis2 - tis1)
# In [15]
# top documents predicted for each question
print(y_pred)


# In [16]
def top_accuracy(y_true, y_pred) -> float:
    right, count = 0, 0
    for y_t in y_true:
        count += 1
        if y_t in y_pred:
            right += 1
    return right / count if count > 0 else 0


# In [17]
acc = top_accuracy(y_test, y_pred)
print('Accuracy:', f'{acc:.4f}')
print('Quantity:', int(acc * len(y_pred)), 'from', len(y_pred))

'''
Discussion
TF-IDF has some problems: 
(1) this algorithm is only able to compute similarity between questions and documents that present the same words, so it can not capture synonyms; 
(2) cannot understand the question context or the meaning of the words.
'''

'''
Word2Vec / Embedding
Word2vec is a technique for natural language processing published in 2013. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence.
word to vector
'''
# In [18]
# create a corpus of tokens
corpus = documents['context'].tolist()
corpus = [preprocess_string(t) for t in corpus]

# In [19]
# train your own model
# vector_size:词向量维度，window：上下文宽度，min_count：为考虑计算的单词的最低词频阈值（即在训练时，会把词频低于1的词去掉）
vectorizer = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=1).wv
'''
seed:默认为1,随机数生成器的种子。每个单词的初始矢量都用单词 + 'str（seed）' 的串联哈希值播种。
请注意，对于完全确定的可重现运行，还必须将模型限制为单个工作线程（“workers=1”），以消除操作系统线程调度中的排序抖动。
（在 Python 3 中，解释器启动之间的可重复性还需要使用“PYTHONHASHSEED”环境变量来控制哈希随机化）。
works:使用多核计算机加快训练速度
'''

# In [20]
# similar words to 'tourist'
print(vectorizer.most_similar('tourist', topn=5))


# In [21]
def transform_text2(vectorizer, text, verbose=False):
    '''
    Transform the text in a vector[Word2Vec]
    vectorizer: sklearn.vectorizer
    text: str
    '''
    tokens = preprocess_string(text)
    words = [vectorizer[w] for w in tokens if w in vectorizer]
    if verbose:
        print('Text:', text)
        print('Vector:', [w for w in tokens if w in vectorizer])
    elif len(words):
        return np.mean(words, axis=0)
    else:
        return np.zeros((300), dtype=np.float32)


# In [22]
# just testing our Word2Vec
print(transform_text2(vectorizer, question, verbose=True))

# In [23]
# let's train the model to retrieve the document id 'c_id'
retriever = NearestNeighbors(**retriever_configs)

# vectorizer the documents, fit the retriever
X = documents['context'].apply(lambda x: transform_text2(vectorizer, x)).tolist()
print(retriever.fit(X, documents['c_id']))

# In [24]
# vectorizer the questions
tis3 = time.perf_counter()
X = data['question'].apply(lambda x: transform_text2(vectorizer, x)).tolist()

# predict one document for each question
y_test = data['c_id']
y_pred = retriever.kneighbors(X, return_distance=False)
tis4 = time.perf_counter()
print(tis4 - tis3)

# In [25]
# top documents predicted for each question
print(y_pred)

# In [26]
acc = top_accuracy(y_test, y_pred)
print('Accuracy:', f'{acc:.4f}')
print('Quantity:', int(acc*len(y_pred)), 'from', len(y_pred))
