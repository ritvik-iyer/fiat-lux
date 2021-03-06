import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import pickle
from os import path
import warnings

warnings.filterwarnings("ignore")


def train_model():
    """
    Data import
    """
    #print("Loading training data...")
    frame = pd.read_csv('model-data.csv')
    """
    Data Preprocessing
    """
    #print("Cleaning data...")
    frame['content'] = frame['content'].apply(preprocess)
    tester = frame.copy()
    tester['num_words'] = tester['content'].apply(get_length)
    max_feature_number = tester.groupby('domain').max()['num_words']['nytimes']
    encodings = [1 if text == 'breitbart' else 0 for text in frame['domain']]
    encoding_map = {1:'Breitbart', 0:'New York Times'}
    data = frame.drop(columns = 'domain')
    data['label'] = encodings
    X = data['content'].values
    y = data['label'].values
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = max_feature_number)
    vectorizer.fit(list(data['content'].values))
    X_cv = vectorizer.transform(X)
    """
    Modeling
    """
    #print("Training model...")
    linsvm_classifier = SGDClassifier(loss = 'hinge', penalty = 'l2', tol = None, max_iter = 1000)
    linsvm_classifier.fit(X_cv, y)
    model_name = 'SGDClassifier.sav'
    vectorizer_name = 'Tf-idfVectorizer.sav'
    pickle.dump(linsvm_classifier, open(model_name, 'wb'))
    pickle.dump(vectorizer, open(vectorizer_name, 'wb'))
    pickle.dump(encoding_map, open('encoding_map.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return

def preprocess(text):
    """ Takes in a string and returns cleaned string"""
    symbols_1 = re.compile('[/(){}\[\]\|@,;]')
    symbols_2 = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))
    text = text.lower()
    text = symbols_1.sub(' ', text)
    text = symbols_2.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in stopwords_set)
    return text

def get_length(text):
    words = text.split(' ')
    return len(words)

def generate_prediction(classifier, vectorizer, filename):
    """Generates one prediction for a given text input using trained classifier and fitted tf-idf vectorizer"""
    with open(filename, 'r', encoding="utf8", errors='ignore') as file:
        data = file.read().replace('\n', '')
    cleaned = preprocess(data)
    word_vector = vectorizer.transform([cleaned])
    return classifier.predict(word_vector)[0]

def getObjFromFile(filename):
    assert path.exists(filename), "The file doesn't exist in the working directory."
    try:
        with open(filename, 'rb') as handle:
            obj = pickle.load(handle)
        return obj
    except Exception as e:
        print(e)
        return

if __name__ == '__main__':
    if not (path.exists('SGDClassifier.sav') and path.exists('Tf-idfVectorizer.sav') and path.exists('encoding_map.pickle')):
        train_model()
    classifier = getObjFromFile('SGDClassifier.sav')
    vectorizer = getObjFromFile('Tf-idfVectorizer.sav')
    encoding_map = getObjFromFile('encoding_map.pickle')
    filename = input('Enter the name of the .txt file containing the article you want to classify. It must be from one of the following outlets: {} \n'.format(', '.join(list(encoding_map.values()))))
    try:
        prediction = generate_prediction(classifier, vectorizer, filename)
        print('The classifier predicts that this article was published by {}.'.format(encoding_map.get(prediction)))
    except Exception as e:
        print('Encountered an error. Please make sure your input is formatted correctly.')
        exit()
