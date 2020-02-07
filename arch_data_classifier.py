#start tika server. The Tika Server is the Parser
#java -jar "path\to\tika-server-1.22.jar"

#import necessary modules
import os
import tika
tika.initVM()
from tika import parser
import os.path
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
from spacy.lang.en import English
nlp = English()
nlp.max_length = 10000000
import nltk
from datetime import datetime
from dateparser.search import search_dates
from gensim.summarization import keywords
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer
import re
import time
from collections import Counter
from summa.summarizer import summarize
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "tesseract.exe"
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import dask.dataframe as dd
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.preprocessing import text

#define a parameter for tika parsers. This declaration solves the status 422 server error
headers = {'X-Tika-PDFextractInlineImages': 'true', "X-Tika-OCRLanguage": "eng"}

#import project data
main_df = pd.read_csv(r"consolidated-training-data.csv", encoding = "ISO-8859-1")
arch_df = main_df[['Arch_Data', 'Path']]

#parse text from scanned files
def ocr_pdf(file):
    images = convert_from_path(file)
    ocr_list = [pytesseract.image_to_string(x) for x in images]
    ocr = ''
    return ocr.join(ocr_list)

#Run Tika Parser on Texts
def return_parsed(paths):
    try:
        return parser.from_file(paths, headers=headers)
    except:
        return 'path error'

def return_texts(parsed, paths):
    if 'content' in parsed and parsed['content'] is not None:
        return parsed['content'] #extract 'content' from parsed texts
    else:
        try:
            return ocr_pdf(paths) #if no 'content' from tika parser, try OCRing the document
        except:
            return "no content"

def remove_whitespace(text):
    return text.strip()

arch_df = dd.from_pandas(arch_df, npartitions=5)
parsed = arch_df.apply(lambda row: return_parsed(row['Path']), axis = 1).compute()
arch_df['Parsed'] = parsed

texts =  arch_df.apply(lambda row: return_texts(row['Parsed'], row['Path']), axis = 1).compute()
arch_df['Texts'] = texts
arch_df = arch_df.compute()

#Drop rows with no text content
no_content = arch_df[arch_df['Texts'] == 'no content'].index
arch_df.drop(no_content, inplace=True)
isnan = arch_df[arch_df['Arch_Data'].isna() == True].index
arch_df.drop(isnan, inplace=True)
arch_df.drop(columns = ['Unnamed: 0']) #only needed if imported from csv

"""Remove Whitespace from Arch Data Values"""
Arch_Data = arch_df.apply(lambda row: remove_whitespace(row['Arch_Data']), axis = 1)
arch_df['Arch_Data'] = Arch_Data

"""Plot of Document Samples Grouped by Arch Data"""
fig = plt.figure(figsize=(8,6))
arch_df.groupby('Arch_Data').Texts.count().plot.bar(ylim=0)
plt.show()

#Import libraries needed for model training and testing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.feature_selection import chi2
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import textstat
from keras.utils.np_utils import to_categorical

#Create Vectorizer Object
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

#create the ML labels by factorizing
y, mappings =arch_df.Arch_Data.factorize()

X = arch_df.Texts

#Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

X_train = tfidf.fit_transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()

#Build, train and test classifier
clf = MultinomialNB(alpha=0.2)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

#5-fold cross-validate and print validation scores
clf_scores = cross_val_score(clf, X, y, cv=10)
print(clf_scores)

# Predict the labels of the test data: y_pred
y_pred = clf.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names = mappings))

#Chi-squared test to see the unigrams and bigrams most correlated with arch documents
features_chi2 = chi2(X, y)
indices = np.argsort(features_chi2[0])
feature_names = np.array(tfidf.get_feature_names())[indices]
unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-20:])))
print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-20:])))

#save trained model
model_save = "arch-model-2.sav"
pickle.dump(clf, open(model_save, 'wb'))

#Open saved trained model and test
loaded_model = pickle.load(open(model_save, 'rb'))
accuracy = loaded_model.score(X_test, y_test)
print(accuracy)
