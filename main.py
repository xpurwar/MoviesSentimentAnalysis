# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
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
            with utils.open(source, 'r') as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.open(source, 'r') as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled

# Define your sources
sources = {
    'testpos.txt': 'TEST_NEG', 
    'testneg.txt': 'TEST_POS', 
    'negnolabel.txt': 'TRAIN_NEG', 
    'posnolabel.txt': 'TRAIN_POS', 
    'train-unsup.txt': 'TRAIN_UNS'
}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=1, window=10, sample=1e-4, negative=5, workers=7)

model.build_vocab(sentences.to_array())

for epoch in range(10):
    model.train(
    sentences.sentences_perm(),
    total_examples=model.corpus_count,  # Total number of sentences
    epochs=model.epochs                 # Number of epochs (default from model)
)

print(model.wv.most_similar('good'))
model['TRAIN_NEG_0']
model.save('./imdb.d2v')
model = Doc2Vec.load('/Users/dineshupadhyay/MoviesSentimentAnalysis/imdb.d2v')