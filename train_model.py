import logging
from gensim.models import word2vec
from gensim.test.utils import common_texts

sentences = word2vec.LineSentence('ptt_QA_seg.txt')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec(sentences, size=300, window=7, workers=10, sg=1, min_count=3, iter=100)

model.save('ptt_QA.word2vec_50.bin')