import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from joblib import dump, load
from collections import defaultdict
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_SGD = 'model/gr/model_sgd.pkl'
MORAL_DATA = 'model/moral-data.csv'

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

def train_model():
    # Preprocessing
    np.random.seed(500)

    df_corpus = pd.read_csv(MORAL_DATA, sep=';')
    df_corpus = df_corpus.drop(['deontic_modality', 'type'], axis=1)
    df_corpus.rename(columns={'general_rule': 'labels'}, inplace=True)

    # Step 1) Remove blank rows if any.
    df_corpus['text'].dropna(inplace=True)

    # Step 2) Change all the text to lower case
    df_corpus['text'] = [entry.lower() for entry in df_corpus['text']]

    # Step 3) Tokenization
    df_corpus['text'] = [word_tokenize(entry) for entry in df_corpus['text']]

    # Step 4) Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    for index, entry in enumerate(df_corpus['text']):
        final_words = []
        lemmatizer = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                final_word = lemmatizer.lemmatize(word, tag_map[tag[0]])
                final_words.append(final_word)
        df_corpus.loc[index,'text_processed'] = str(final_words)

    # train model
    sgd = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None))
    ])
    sgd.fit(df_corpus['text_processed'], df_corpus['labels'])
    dump(sgd, MODEL_SGD) 

    return sgd

def preprocess(text):
    text = text.lower()
    text = word_tokenize(text)
    
    final_words = []
    lemmatizer = WordNetLemmatizer()
    for word, tag in pos_tag(text):
        if word not in stopwords.words('english') and word.isalpha():
            final_word = lemmatizer.lemmatize(word, tag_map[tag[0]])
            final_words.append(final_word)
    return str(final_words)

def predict_class(sentence):
    try:
        clf = load(MODEL_SGD) 
    except FileNotFoundError:
        logger.warning('Could not load SGD model.')
        return None
    
    text = [preprocess(sentence)]
    return clf.predict(text)[0], max(clf.predict_proba(text)[0])