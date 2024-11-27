# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

# random
import random

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled
    
sources = {'testpos.txt':'TEST_NEG', 'testneg.txt':'TEST_POS', 'negnolabel.txt':'TRAIN_NEG', 'posnolabel.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

sentences = LabeledLineSentence(sources)



# import logging
# import gensim.models
# from gensim import utils
# from gensim.test.utils import datapath
# import numpy as np    

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



# class MyCorpus:
#     """An iterator that yields sentences (lists of str)."""

#     def __iter__(self):
#         corpus_path = datapath('/Users/dineshupadhyay/MoviesSentimentAnalysis/movies.txt')
#         for line in open(corpus_path):
#             # assume there's one document per line, tokens separated by whitespace
#             yield utils.simple_preprocess(line)

# sentences = MyCorpus()
# model = gensim.models.Word2Vec(sentences=sentences)
# # model.build_vocab(sentences)  # Build the vocabulary
# # model.train(sentences, total_examples=model.corpus_count, epochs=10)  

# print("Hello")
# # gensim will delete any word that doesn't appear more than 5 times
# print(model.wv.most_similar(positive=['car', 'vehicle'], topn=5))
# vec_king = model.wv['king']
# print(vec_king)

# model.wv.save_word2vec_format('model.bin', binary=False)

# from gensim.models import KeyedVectors
# kv = KeyedVectors(512)
# kv.add(model.words, model.wv)
# kv.save(model.kvmodel)

# # trying out smaller toy corpus
# sentences = [['i', 'like', 'apple', 'pie', 'for', 'dessert'],
#            ['i', 'dont', 'drive', 'fast', 'cars'],
#            ['data', 'science', 'is', 'fun'],
#            ['chocolate', 'is', 'my', 'favorite'],
#            ['my', 'favorite', 'movie', 'is', 'predator']]