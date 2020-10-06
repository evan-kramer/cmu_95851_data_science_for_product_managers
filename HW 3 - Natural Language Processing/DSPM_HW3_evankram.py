# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:46:30 2020

@author: evan.kramer
"""
# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

# You'll be creating models to predict the reviews that lead to recommendations 
# for the product, 5-star ratings, and reviews considered helpful, which are found:
# in the field reviews.doRecommend, reviews.rating, and reviews.numHelpful respectively.
# Create summary statistics and histograms for each of these fields. Do you see any issues 
# in using these fields as outcome (target) variables? (3 points)
kindle_reviews = pd.read_csv('kindle_reviews.csv', encoding = 'cp1252') # needed to change encoding because the file was not UTF-8 encoded
kindle_reviews.isnull().sum()
kindle_reviews[['reviews.doRecommend', 'reviews.rating', 'reviews.numHelpful']].describe()
kindle_reviews[['reviews.doRecommend', 'reviews.rating', 'reviews.numHelpful']].info()
kindle_reviews['reviews.doRecommend'].value_counts()
for c in ['reviews.rating', 'reviews.numHelpful']:
    print(kindle_reviews[c].value_counts())
    plt.hist(kindle_reviews[c])
    plt.show()
plt.bar(x = np.array([True, False]),
        height = kindle_reviews['reviews.doRecommend'].value_counts(),
        tick_label = np.array([True, False]))
plt.title('reviews.doRecommend')
plt.show()
kindle_reviews['reviews.fiveStarRating'] = kindle_reviews['reviews.rating'] == 5
folds = 5
kindle_reviews['fold'] = np.random.randint(low = 0, high = folds-1, size = len(kindle_reviews))

# Up-sample ratings
pd.concat([kindle_reviews[kindle_reviews['reviews.fiveStarRating'] == False], 
           resample(kindle_reviews[kindle_reviews['reviews.fiveStarRating'] == True],
                    replace = False,
                    n_samples = len(kindle_reviews[kindle_reviews['reviews.fiveStarRating'] == False]),
                    random_state = 1234)])['reviews.fiveStarRating'].value_counts()
          

# Prepare the text of the reviews in the reviews.text field for analysis by eliminating
# stopwords. What are the top 10 most frequent words? What are the top 10 nouns? 
# What are the top 10 adjectives? (3 points)
import nltk
from collections import Counter
import re
text = kindle_reviews['reviews.text']
cleaned_text = []
stop_words = nltk.corpus.stopwords.words('english') + ['amazon', 'kindle', "'s", 
                                                       'voyage', 'nt']

# Remove stopwords and stem/lemmatize
def stemmed_sentence(sentence):
    # Tokenize sentence
    tokenized_words = nltk.tokenize.word_tokenize(sentence.lower())
    # Identify part of speech https://www.nltk.org/book/ch05.html
    pos = nltk.pos_tag(tokenized_words)
    # Initialize empty list
    stemmed_sentence = []
    # Loop
    for word in tokenized_words:
        # Ignore/remove stopwords
        if word in stop_words or len(word) < 2: 
            pass
        else:
            # Use Porter stemming method (or lemmatize?)
            word_clean = nltk.stem.PorterStemmer().stem(re.sub("[\.\-']", '', word))
            # Lemmatize
            word_clean = nltk.stem.WordNetLemmatizer().lemmatize(word_clean)
            # Append to list
            stemmed_sentence.append(word_clean)
            stemmed_sentence.append(' ')
    return ''.join(stemmed_sentence)

kindle_reviews['reviews.textClean'] = [stemmed_sentence(kindle_reviews['reviews.text'][t]) for t in range(len(kindle_reviews['reviews.text']))]
words = (kindle_reviews['reviews.textClean']
         .str.lower()
         .str.cat()
         .split()
)

pd.DataFrame(Counter(words).most_common(10),
                    columns=['Word', 'Freq.']).set_index('Word')

# Tag by part of speech
part_of_speech = pd.DataFrame()
for i in range(len(kindle_reviews)): 
    tokenized_words = nltk.tokenize.word_tokenize(kindle_reviews['reviews.textClean'][i].lower())
    pos = nltk.pos_tag(tokenized_words)
    part_of_speech = pd.concat([part_of_speech, 
                                pd.DataFrame({'word': [pos[i][0].strip() for i in range(len(pos))],
                                              'part_of_speech': [pos[i][1].strip() for i in range(len(pos))]})])

# Top total
part_of_speech.value_counts().head(10) # POS definitions here: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

# Top nouns
(part_of_speech[part_of_speech.part_of_speech.str.contains('^N') == True].
 value_counts().head(10))

# Top adjectives
(part_of_speech[part_of_speech.part_of_speech.str.contains('^J') == True].
 value_counts().head(10))    

# What are the top 10 most frequent words in reviews that do not recommend purchase
# of the Kindle? (3 points)
top10_words_no_rec = (kindle_reviews[kindle_reviews['reviews.doRecommend'] == False]['reviews.textClean']
                      .str.lower()
                      .str.cat()
                      .split())
display(pd.DataFrame(Counter(top10_words_no_rec).most_common(10),
                     columns=['Word', 'Freq']).set_index('Word'))


# Create a model that will predict when a customer will give the Kindle a 5-star
# rating based on the text of the review. Evaluate the accuracy of your model. (3 points)
import random
np.random.seed(123)

# Create nested list of tokenized words and whether the text was for a 5-star review
documents = [(nltk.tokenize.word_tokenize(kindle_reviews['reviews.textClean'][i]),
              kindle_reviews['reviews.fiveStarRating'][i]) 
              for i in range(len(kindle_reviews))]
random.shuffle(documents) # randomly reorder to create training/test splits

# Create list of distinct words in reviews.textClean
word_features = (list(set(kindle_reviews['reviews.textClean'].str.cat().split())))

# Define function to extract features
def document_features(document): 
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Define feature sets
featuresets = [(document_features(d), c) for (d,c) in documents]

# Create training and test set and train model
split = int(len(featuresets) * 0.6)
train_set, test_set = featuresets[split:], featuresets[:split]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Calculate model accuracy and determine most informative features
print('Model accuracy:', nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(10)

# Retry with up-sampled and down-sampled datasets
upsampled = pd.merge(kindle_reviews_upsampled_ratings,
                     kindle_reviews[['reviews.textClean']],
                     how = 'left', left_index = True, right_index = True)
downsampled = pd.merge(kindle_reviews_downsampled_ratings,
                      kindle_reviews[['reviews.textClean']],
                      how = 'left', left_index = True, right_index = True)
# Upsampled
documents = [(nltk.tokenize.word_tokenize(upsampled['reviews.textClean'][i]),
              upsampled['reviews.fiveStarRating'][i]) 
             for i in range(len(kindle_reviews))]
random.shuffle(documents) # randomly reorder to create training/test splits
word_features = (list(set(upsampled['reviews.textClean'].str.cat().split())))
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[split:], featuresets[:split]
classifier = nltk.NaiveBayesClassifier.train(train_set)
# Re-calculate model accuracy and determine most informative features
print('Model accuracy (upsampled):', nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(10)
# Downsampled
documents = [(nltk.tokenize.word_tokenize(downsampled['reviews.textClean'][i]),
              downsampled['reviews.fiveStarRating'][i]) 
             for i in range(len(kindle_reviews))]
random.shuffle(documents) # randomly reorder to create training/test splits
word_features = (list(set(downsampled['reviews.textClean'].str.cat().split())))
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[split:], featuresets[:split]
classifier = nltk.NaiveBayesClassifier.train(train_set)
# Re-calculate model accuracy and determine most informative features
print('Model accuracy (downsampled):', nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(10)


'''
import random
from nltk.corpus import brown
tagged_sents = list(brown.tagged_sents(categories='news'))
random.shuffle(tagged_sents)
size = int(len(tagged_sents) * 0.1)
train_set, test_set = tagged_sents[size:], tagged_sents[:size]
file_ids = brown.fileids(categories='news')
size = int(len(file_ids) * 0.1)
train_set = brown.tagged_sents(file_ids[size:])
test_set = brown.tagged_sents(file_ids[:size])
train_set = brown.tagged_sents(categories='news')
test_set = brown.tagged_sents(categories='fiction')
def tag_list(tagged_sents):
    return [tag for sent in tagged_sents for (word, tag) in sent]
def apply_tagger(tagger, corpus):
    return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]
gold = tag_list(brown.tagged_sents(categories='editorial'))
test = tag_list(apply_tagger(t2, brown.tagged_sents(categories='editorial')))
cm = nltk.ConfusionMatrix(gold, test)
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
'''

# Create a model that will predict when at least two customers will find a review
# helpful. Evaluate the accuracy of your model. (3 points)
