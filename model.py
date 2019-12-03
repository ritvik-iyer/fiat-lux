import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import warnings

warnings.filterwarnings("ignore")

"""
Data import
"""
print("Reading training data...")
frame = pd.read_csv('model-data.csv')

"""
Data Preprocessing
"""
print("Cleaning data...")
symbols_1 = re.compile('[/(){}\[\]\|@,;]')
symbols_2 = re.compile('[^0-9a-z #+_]')
stopwords_set = set(stopwords.words('english'))

def preprocess(text):
    """ Takes in a string and returns cleaned string"""
    text = text.lower()
    text = symbols_1.sub(' ', text)
    text = symbols_2.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in stopwords_set)
    return text

frame['content'] = frame['content'].apply(preprocess)

def get_length(text):
    words = text.split(' ')
    return len(words)

tester = frame.copy()
tester['num_words'] = tester['content'].apply(get_length)

encodings = [1 if text == 'breitbart' else 0 for text in frame['domain']]
data = frame.drop(columns = 'domain')
data['label'] = encodings

X = data['content'].values
y = data['label'].values
vectorizer = TfidfVectorizer(stop_words = 'english', max_features = tester.groupby('domain').max()['num_words']['nytimes'])
vectorizer.fit(list(frame['content'].values))
X_cv = vectorizer.transform(X)

"""
Modeling
"""
print("Training model...")
linsvm_classifier = SGDClassifier(loss = 'hinge', penalty = 'l2', tol = None, max_iter = 1000)
linsvm_classifier.fit(X_cv, y)

"""
Shell
"""
print()
valid_articles = {0: "NY Times", 1: "Breitbart"}
while True:
    user_input = input("Enter the text an article from one of the following outlets: {}".format(valid_articles.values()))
    try:
        user_vector = vectorizer.transform([preprocess(user_input)])
        print("Classifier prediction: ", valid_articles[linsvm_classifier.predict(user_vector)[0]])
    except Exception as e:
        print("Encountered error. Exiting shell.")
        exit()
