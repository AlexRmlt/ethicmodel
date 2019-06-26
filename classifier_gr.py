import pandas as pd

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from joblib import dump, load
from collections import defaultdict
import os
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

CLASSIFIER = 'svm'

MODEL_GR = os.path.join(os.path.dirname(__file__), 'model', 'gr', 'model_gr.pkl') 
MORAL_DATA = os.path.join(os.path.dirname(__file__), 'model', 'moral-data.csv')
NAMES_LIST = os.path.join(os.path.dirname(__file__), 'model', 'names.txt')

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

def train_model():
    # Preprocessing
    try:
        df_corpus = pd.read_csv(MORAL_DATA, sep=';')
    except FileNotFoundError:
        logger.warning('Could not train model: did not find moral data csv.')
        raise

    df_corpus = df_corpus.drop(['deontic_modality', 'type'], axis=1)
    df_corpus.rename(columns={'general_rule': 'labels'}, inplace=True)

    # Remove blank rows if any.
    df_corpus['text'].dropna(inplace=True)

    # Tokenization, remove stop words and names, perfom word stemming/lemmenting.
    for index, entry in enumerate(df_corpus['text']):
        df_corpus.loc[index,'text_processed'] = preprocess(entry)

    # train model
    cls = classifier(CLASSIFIER)
    cls.fit(df_corpus['text_processed'], df_corpus['labels'])

    # create directory if it does not exist
    directory = MODEL_GR.rsplit('/', 1)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    dump(cls, MODEL_GR) 

    return cls

def classifier(name):
    if name == 'sgd':
        cls = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None))
        ])
    elif name == 'svm':
        cls = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)),
        ])
    elif name == 'logreg':
        cls = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(n_jobs=1, C=1e5)),
        ])
    else:
        cls = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None))
        ])

    return cls

def preprocess(text):
    text = text.lower()
    text = word_tokenize(text)

    with open(NAMES_LIST) as file:
        names = set(file.read().split('\n'))
    
    final_words = []
    lemmatizer = WordNetLemmatizer()

    try:
        pos_tags = pos_tag(text)
    except MemoryError:
        for word in text:
            if word not in stopwords.words('english') and word not in names and word.isalpha():
                final_word = lemmatizer.lemmatize(word)
                final_words.append(final_word)
    else:
        for word, tag in pos_tags:
            if word not in stopwords.words('english') and word not in names and word.isalpha():
                final_word = lemmatizer.lemmatize(word, tag_map[tag[0]])
                final_words.append(final_word)
    
    return str(final_words)

def predict_class(sentence):
    try:
        clf = load(MODEL_GR) 
    except FileNotFoundError:
        logger.warning('Could not load SGD model.')
        return None, None
    
    text = [preprocess(sentence)]
    return clf.predict(text)[0], max(clf.predict_proba(text)[0])