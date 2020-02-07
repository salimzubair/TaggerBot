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
pytesseract.pytesseract.tesseract_cmd = r"tesseract.exe"
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import dask.dataframe as dd
import matplotlib.pyplot as plt

#define a parameter for tika parsers. This declaration solves the status 422 server error
headers = {'X-Tika-PDFextractInlineImages': 'true', "X-Tika-OCRLanguage": "eng"}

#import project data
main_df = pd.read_csv(r"consolidated-training-data.csv", encoding = "ISO-8859-1")
doc_type_df = main_df[['Document_Type', 'Path']]

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

#Extract Text Content of Parsed Documents. If Documents not parsed, OCR the document
def return_texts(parsed, paths):
    if 'content' in parsed and parsed['content'] is not None:
        return parsed['content'] #extract 'content' from parsed texts
    else:
        try:
            return ocr_pdf(paths) #if no 'content' from tika parser, try OCRing the document
        except:
            return "no content"

#Function to remove whitespaces from 
def remove_whitespace(text):
    return text.strip()

#Apply function to parse documents
doc_type_df = dd.from_pandas(doc_type_df, npartitions=5)
parsed = doc_type_df.apply(lambda row: return_parsed(row['Path']), axis = 1).compute()
doc_type_df['Parsed'] = parsed

#Apply function to retrieve text or OCR documents
texts =  doc_type_df.apply(lambda row: return_texts(row['Parsed'], row['Path']), axis = 1).compute()
doc_type_df['Texts'] = texts
doc_type_df = doc_type_df.compute()

#Apply function to remove whitespace from Document_Type field
Document_Type = doc_type_df.apply(lambda row: remove_whitespace(row['Document_Type']), axis = 1)
doc_type_df['Document_Type'] = Document_Type

#Drop rows with no text content
no_content = doc_type_df[doc_type_df['Texts'] == 'no content'].index
doc_type_df = doc_type_df.drop(columns = no_content, inplace=True)
doc_type_df.drop(columns = ['Unnamed: 0']) #only needed if imported from csv
doc_type_df.head()

#Save DataFrame as csv file
doc_type_df.to_csv("parsed-doc-type-training-data-3.csv")

"""Plot of Document Samples Grouped by Document Type"""
fig = plt.figure(figsize=(8,6))
doc_type_df.groupby('Document_Type').Texts.count().plot.bar(ylim=0)
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

#Create Vectorizer Object
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

#create the ML labels by factorizing
y, mappings =doc_type_df.Document_Type.factorize()

#Create ML feature
X = doc_type_df.Texts

#Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

#Vectorize Text
X_train = tfidf.fit_transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()

#Build, train and test classifier
clf = SVC(C=10, gamma = 0.1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

#5-fold cross-validate and print validation scores
cv_scores = cross_val_score(clf, X, y, cv=5)

# Predict the labels of the test data: y_pred
y_pred = clf.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names = mappings))

#Print most correlated unigrams and bigrams per document type
N = 10
new_df = doc_type_df
new_df['category_id'] = new_df['Document_Type'].factorize()[0]
category_df = doc_type_df[['Document_Type', 'category_id']]
category_to_id = dict(category_df.values)
for Product, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(X, y == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

#save trained model
model_save = r"doc-type-model.sav"
pickle.dump(clf, open(model_save, 'wb'))

#Open saved trained model and test
loaded_model = pickle.load(open(model_save, 'rb'))
accuracy = loaded_model.score(X_test, y_test)
print(accuracy)