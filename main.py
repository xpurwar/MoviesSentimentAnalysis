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

# Defining our sources
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
model = Doc2Vec.load('D:/cs159/MoviesSentimentAnalysis/imdb.d2v')

# checking keys
print(list(model.dv.index_to_key)[:])  # Print the first 10 keys
print(f"Number of keys in model: {len(model.dv.index_to_key)}")




# classification and evaluation
# train all 100 words, initialized as np arrays
train_arrays = numpy.zeros((200, 100)) 
train_labels = numpy.zeros(200)

for i in range(100):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)

    train_arrays[i] = model[prefix_train_pos]
    train_arrays[100 + i] = model[prefix_train_neg]  
    train_labels[i] = 1                              # Positive class
    train_labels[100 + i] = 0            

print (train_arrays)
print (train_labels)

# extract results from test data - assigning classifiers!
test_arrays = numpy.zeros((200, 100))
test_labels = numpy.zeros(100)
print("YUHHH /n/n\n\n")
# Adjust test arrays for only 4 positive and 4 negative samples
num_test_samples = 4  # Number of positive and negative test samples
total_test_samples = num_test_samples * 2  # Total test samples (positive + negative)

test_arrays = numpy.zeros((total_test_samples, 100))
test_labels = numpy.zeros(total_test_samples)

# Populate test data
for i in range(num_test_samples):
    # Positive samples
    prefix_test_pos = 'TEST_POS_' + str(i)
    test_arrays[i] = model[prefix_test_pos]
    test_labels[i] = 1  # Label as positive

    # Negative samples
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[num_test_samples + i] = model[prefix_test_neg]
    test_labels[num_test_samples + i] = 0  # Label as negative

print(test_arrays)
print(test_labels)
# Debug shapes
print("Train arrays shape:", train_arrays.shape)  # Should be (200, 100)
print("Train labels shape:", train_labels.shape)  # Should be (200,)
print("Test arrays shape:", test_arrays.shape)    # Should be (8, 100)
print("Test labels shape:", test_labels.shape)    # Should be (8,)


# # training a logistic regression classifier on our training data
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)  # Train with 200 samples

# Evaluate on 8 test samples
accuracy = classifier.score(test_arrays, test_labels)
print("Test Accuracy:", accuracy)