# -*- coding: utf-8 -*-

import numpy as np
from scipy import spatial
from gensim.models import word2vec
import jieba
import re

def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    # print(words)
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words> 0):
        feature_vec = np.divide(feature_vec, n_words)

    return feature_vec

def compute(sentence, model, index2word_set, stopword_set) :
    lines = sentence.split('\t')
    Question = ''
    ans = ''
    res = []

    words = list(jieba.cut(lines[0], cut_all=False))
    if len(words) == 0 :
        # print('---------------------------------------------------')
        Question = lines[0]
    else :
        for word in words:
            if word not in stopword_set:
                Question = Question + word + ' '
        s1_afv = avg_feature_vector(Question, model=model, num_features=300, index2word_set=index2word_set)
        lines.pop(0)

    for line in lines :
        s = line.split(')')[1]
        words = list(jieba.cut(s, cut_all=False))
        if len(words) == 0 :
            # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            ans = line
        else :
            for word in words:
                if word not in stopword_set:
                    ans = ans + word + ' '

        s2_afv = avg_feature_vector(ans, model=model, num_features=300, index2word_set=index2word_set)
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
        resultInfo = {'id' : lines.index(line) + 1, 'score': sim, 'text': line}
        res.append(resultInfo)
        ans = ''
    res.sort(key=lambda x:x['score'], reverse=True)
    return str(res[0]['id'])

def main():
    # set stop words
    jieba.set_dictionary('extra_dict/dict.txt.big')
    stopword_set = set()
    with open('extra_dict/stop_words.txt', 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
    
    # use model
    model = 'ptt_QA.word2vec_50.bin'
    model_w2v = word2vec.Word2Vec.load(model)
    index2word_set = set(model_w2v.wv.index2word)
    ans = []
    count = 0

    output = open('answer.csv', 'w', encoding='utf-8')

    with open('question.txt', 'r', encoding='utf-8') as ssss:
        ssss = list(ssss)
        # print(ssss)
        for lin in ssss :
            line = lin.strip('\n').strip()
            result = compute(sentence=line, model=model_w2v, index2word_set=index2word_set, stopword_set=stopword_set)
            output.write(result)
            output.write('\n')

if __name__ == "__main__":
    main()
