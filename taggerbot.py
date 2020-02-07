#start tika server. The Tika Server is the Parser
!java -jar "path\to\tika-server-1.22.jar"

#import necessary modules
import os
import tika
tika.initVM()
from tika import parser
import os.path
import pandas as pd
import numpy as np
import spacy
import dask.dataframe as dd
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
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

#define a parameter for tika parsers. This declaration solves the status 422 server error
headers = {'X-Tika-PDFextractInlineImages': 'true', "X-Tika-OCRLanguage": "eng"}

"""Importing all Lookup Data"""

#import project lookup data
proj_lookup = pd.read_csv("proj_codes.csv", encoding = "ISO-8859-1")

#import circuit lookup table and convert to list
circuit_lookup = list(pd.read_csv("circuits.csv", header=None, encoding = "ISO-8859-1")[0])

#import station lookups
code_lookup = list(pd.read_csv("station-codes.csv", header = None)[0])
code_name_lookup = list(pd.read_csv("station-names.csv", encoding = "ISO-8859-1")['Code'])
name_lookup = list(pd.read_csv("station-names.csv", encoding = "ISO-8859-1")['Name'])


#import reservoirs lookup and convert to lists
reservoir_name = list(pd.read_csv("reservoirs.csv", encoding = "ISO-8859-1", header = None)[0])
reservoir_values = list(pd.read_csv("reservoirs.csv", encoding = "ISO-8859-1", header = None)[1])

#parse text from scanned files
def ocr_pdf(file):
    images = convert_from_path(file)
    ocr_list = [pytesseract.image_to_string(x) for x in images]
    ocr = ''
    return ocr.join(ocr_list)

"""RETURN THE ROOT FOLDER BY EXTRACTING MID STRING"""
def find_root(path):
    return path[32: path.find("\\", 32)] #the number here is 32 because raw strings. With regular strings the number is 26

"""Function to extract dates from a piece of text"""
today = datetime.today()

#Match strings that contain only numbers and special characters
def special_match(strg, search=re.compile(r'[^|\&+\-%*/=!>0-9.]').search):
    return not bool(search(strg))

def find_dates(text):
    found_dates = [*datefinder.find_dates(text, source = True)]
    """Exclude Possible error matches"""
    found_dates = [x for x in found_dates if x[0] <= today and x[0] >= datetime(1940, 1, 1)] #only return dates between 1940 and today
    found_dates = [x for x in found_dates if re.fullmatch('\d{4}', x[1]) is None]  #search_dates sometimes confuses years for full dates. This excludes those errors
    found_dates = [x for x in found_dates if special_match(x[1]) is True or len([ele for ele in month_list if(ele in x[1].lower())]) != 0] #only return matches that contain a month string or just digits and special characters
    return found_dates[0][0] #only return datetime object of first match

# Function to preprocess text
def preprocess(text):
    # Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]
    
    return ' '.join(a_lemmas)

#collect all the file paths in folder and return as numpy array
def return_paths(folder):
    paths = np.array([os.path.join(r, file) for r, d, f in os.walk(folder) for file in f])
    return paths

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

def return_zip_items(paths, texts):
    return ''.join([paths, texts])

def return_metadata(parsed):
    try:
        return parsed['metadata']
    except:
        return "no content"

def return_filenames(paths):
    try:
        return os.path.basename(paths)
    except:
        return "no content"

def return_dates(filenames, texts, metadata):
    try:
        return find_dates(filenames)
    except:
        try:
            return find_dates(texts)
        except:
            try:
                return metadata['Last-Modified']
            except:
                return "no content"

"""Extract Keywords from text"""
def return_keywords(texts):
    xkeywords = []
    values = keywords(text=preprocess(texts),split='\n',scores=True)
    for x in values[:20]:
        xkeywords.append(x[0])
    try:
        return xkeywords 
    except:
        return "no content"

#determine the document author
def return_authors(paths, metadata):
    authors = []
    try:    
        authors.append(find_root(repr(paths)))
        if metadata is not "no content" and metadata is not None and 'Author' in metadata:
            authors.append(metadata['Author'])
        if metadata is not "no content" and metadata is not None and 'Company' in metadata:
            authors.append(metadata['Company'])    
        return authors
    except:
        return 'no content'

def return_descriptions(texts):
    try:
        return summarize(texts, words=50, language='english')
    except:
        return 'no content'

def return_proj_ids(zip_items):
    #populate proj_codes list with all project IDs found
    proj_no = []
    proj_id_regex = r"[BDEFGTY][ABCEFILMPRSVYZ][-\s]?\d{4,5}|[A-Z]{3}MON[-\s]?\d{1,2}|[A-Z]{3}WORKS[-\s]?\d{1,2}|[A-Z]{3}MON[-\s]?\d{1,2}[A-Z]|[A-Z]{3}WORKS[-\s]?\d{1,2}[A-Z]"
    proj_no.append(re.findall(proj_id_regex , zip_items[:1000]))

    #remove duplicate occurrences
    proj_codes = [list(dict.fromkeys(x)) for x in proj_no]
        
    #reformat wrongly formatted project codes to standard
    regex1 = r"([A-Z]{3}MON)\s?(\d{1,2}[A-Z]?)"           #ABFMON 02(A)/ ABFMON02(A) ----> ABFMON-02(A)
    regex2 = r"([A-Z]{3}WORKS)\s?(\d{1,2}[A-Z]?)"         #BRGWORKS 01(A)/BRGWORKS01(A)  ---->  BRGWORKS-01(A)
    regex3 = r"([A-Z]{3}MON)-(\d{1}[A-Z]?)"               #ABFMON-2/ CLBMON-1A  ---->  ABFMON-02(A)
    regex4 = r"([A-Z]{3}WORKS)-(\d{1}[A-Z]?)"             #BRGWORKS-1/ CLBWORKS-2A  ---->  BRGWORKS-01(A)
    regex5 = r"([BDEFGTY][ABCEFILMPRSVYZ])\s?(\d{4,5})"   #TY7111 / TY 7111  ---->  TY-7111

    #capture rightly formatted project codes too
    regex6 = r"[A-Z]{3}MON-\d{2}[A-Z]?"
    regex7 = r"[A-Z]{3}WORKS-\d{2}[A-Z]?"
    regex8 = r"[A-Z]{2}-\d{4}"

    new_proj = []
    for reg in proj_codes:
        a = [re.sub(regex1, r"\1-\2", x) for x in reg]
        b = [re.sub(regex2, r"\1-\2", x) for x in reg] + a
        c = [re.sub(regex3, r"\1-0\2", x) for x in b + reg] + b
        d = [re.sub(regex4, r"\1-0\2", x) for x in c + reg] + c
        e = [re.sub(regex5, r"\1-\2", x) for x in reg] + d
        e.append(re.findall(regex6 , str(reg)))
        e.append(re.findall(regex7 , str(reg)))
        e.append(re.findall(regex8 , str(reg)))
        new_proj.append(e)
    proj_codes = new_proj
    
    #validate project codes are in project codes lookup table
    validate_codes = []
    for x in proj_codes:
        y = [z for z in x if z in list(proj_lookup['Project Code'])]
        validate_codes.append(y)
    proj_codes = []

    #remove duplicate occurrences      
    for x in validate_codes:
        proj_codes = list(dict.fromkeys(x))
    return proj_codes

#Use proj_lookup to find project name for all project codes
def return_proj_names(proj_ids):
    try:
        for x in proj_ids:
            proj_names = [list(proj_lookup['Project Name'])[list(proj_lookup['Project Code']).index(y)] for y in proj_ids]
        return proj_names
    except:
        return "no content"

def return_circuits(zip_items):
    try:
        item_delimited = re.split('[|\^&+\-%*/=!>\s\-.,\\\\]', zip_items)     #delimit because some circuits are contained in other circuits e.g. 1L1 AND 1L135 
        circuit_list = [circuit for circuit in circuit_lookup if circuit in item_delimited]
        return circuit_list
    except:
        return 'no content'

def return_stations(zip_items):
    #find station names in each document 
    name_list = []
    name_list.append([code_name_lookup[name_lookup.index(name)] for name in name_lookup if name in zip_items.lower()])

    #remove duplicate occurrences
    for x in name_list:
        stations = list(dict.fromkeys(x))
    return stations

def return_reservoirs(zip_items):
    reservoirs = []
    #find reservoirs in paths and texts
    reservoirs = [reservoir_name[reservoir_values.index(value)] for value in reservoir_values if value in zip_items.lower()]
    return reservoirs

def tagger(inputFolder):
    
    paths = return_paths(inputFolder)
    tagger_df = pd.DataFrame({'Paths': paths})
    tagger_df = dd.from_pandas(tagger_df, npartitions=5) #Convert Pandas DataFrame to Dask DataFrame
    
    parsed = tagger_df.apply(lambda row: return_parsed(row['Paths']), axis = 1)
    tagger_df['Parsed'] = parsed 
    
    texts =  tagger_df.apply(lambda row: return_texts(row['Parsed'], row['Paths']), axis = 1)
    tagger_df['Texts'] = texts
    
    zip_items =  tagger_df.apply(lambda row: return_zip_items(row['Paths'], row['Texts']), axis = 1)
    tagger_df['Zip Items'] = zip_items
                                                          
    metadata =  tagger_df.apply(lambda row: return_metadata(row['Parsed']), axis = 1)
    tagger_df['Metadata'] = metadata
    
    filenames =  tagger_df.apply(lambda row: return_filenames(row['Paths']), axis = 1)
    tagger_df['Filenames'] = filenames
    
    dates =  tagger_df.apply(lambda row: return_dates(row['Filenames'], row['Texts'], row['Metadata']), axis = 1)
    tagger_df['Dates'] = dates
    
    keywords_list =  tagger_df.apply(lambda row: return_keywords(row['Texts']), axis = 1)
    tagger_df['Keywords'] = keywords_list
    
    authors =  tagger_df.apply(lambda row: return_authors(row['Paths'], row['Metadata']), axis = 1)
    tagger_df['Authors'] = authors
    
    descriptions =  tagger_df.apply(lambda row: return_descriptions(row['Texts']), axis = 1)
    tagger_df['Descriptions'] = descriptions
    
    proj_ids =  tagger_df.apply(lambda row: return_proj_ids(row['Zip Items']), axis = 1)
    tagger_df['Project IDs'] = proj_ids
    
    proj_names =  tagger_df.apply(lambda row: return_proj_names(row['Project IDs']), axis = 1)
    tagger_df['Project Names'] = proj_names
    
    circuits =  tagger_df.apply(lambda row: return_circuits(row['Zip Items']), axis = 1)
    tagger_df['Circuits'] = circuits
    
    stations =  tagger_df.apply(lambda row: return_stations(row['Zip Items']), axis = 1)
    tagger_df['Stations'] = stations
    
    reservoirs =  tagger_df.apply(lambda row: return_reservoirs(row['Zip Items']), axis = 1)
    tagger_df['Reservoirs'] = reservoirs
    tagger_df = tagger_df.compute() #Convert Dask DataFrame back to Pandas DataFrame #Exploring faster options
    
    return tagger_df
