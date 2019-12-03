import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings("ignore")


def train_model():
    """
    Data import
    """
    print("Loading training data...")
    frame = pd.read_csv('model-data.csv')
    """
    Data Preprocessing
    """
    print("Cleaning data...")
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
    print("Training model...")
    linsvm_classifier = SGDClassifier(loss = 'hinge', penalty = 'l2', tol = None, max_iter = 1000)
    linsvm_classifier.fit(X_cv, y)
    return linsvm_classifier, vectorizer, encoding_map

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

def generate_prediction(classifier, vectorizer, input_data):
    """Generates one prediction for a given text input using trained classifier and fitted tf-idf vectorizer"""
    cleaned = preprocess(input_data)
    word_vector = vectorizer.transform([cleaned])
    return classifier.predict(word_vector)[0]

if __name__ == '__main__':
    classifier, vectorizer, encoding_map = train_model()
    user_input = input('Enter the text of an article from one of the following outlets: {} \n'.format(', '.join(list(encoding_map.values()))))
    try:
        prediction = generate_prediction(classifier, vectorizer, user_input)
        print('The classifier predicts that this article was published by {}'.format(encoding_map.get(prediction)))
    except Exception as e:
        print('Encountered an error. Please make sure your input is formatted correctly.')
        exit()
