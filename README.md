# TaggerBot

TaggerBot is a machine learning application that reads unstructured documents and applies Natural Language Processing (NLP) and Natural Language Understanding (NLU) techniques to provide summaries, metadata and structure to files in document repositories. TaggerBot adds context to otherwise meaningless collections of unstructured information and assists users in information retrieval.

TaggerBot is developed for enterprise knowledge discovery and document classification. TaggerBot classifies whether a document is useful to keep or transitory, it also determines the following:

- The document type
- The subject of the document
- Whether or not the document contains sensitive data
- A summary of the document
- Top keywords in the document
- The project ID and project name the document relates to
- Geographical features and assets related to the document

TaggerBot determines all the above by parsing the text within the document and using machine learning algorithms and rule-based programming to determine the apprpriate metadata tags. Note that TaggerBot was trained specifically for this corporation and therefore, would need to be trained with new data for other companies and use cases.


## Metadata

1. **Document Type**: TaggerBot uses a Multilayer Perceptron Classifier algorithm to determine the document type of the document.The document type is a signle label field with 11 classes.

2. **Subject Area**: TaggerBot uses a Linear Support Vector Classifier algorithm to determine the subject area of the document. The subject area field is a mult-label field with 8 classes.

3. **Contains Sensitive Data**: TaggerBot uses a Linear Support Vector Classifier algorithm to determine whether or not a document contains sensitive data. In this case, sensitive data is archaeological information. This is a binary class field. 

4. **Document Summary**: Taggerbot uses an extractive text summarization algorithm to give a brief summary of all documents.

5. **Keywords**: Taggerbot uses an unsupervised model to extract top keywords in a document based on term frequency.

6. **Project IDs and Project Names**: Given a complete list of project IDs and project names, TaggerBot parses the text in documents and cross-refences the content against the list of project IDs and project names to tag documents to their applicable projects.

7. **Geographical assets**: My client had a requirement to tag documents to certain geographical assets such as stations, reservoirs, circuits, offices, etc. where applicable. Given the complete list of such geographical assets, TaggerBot tags each document to the applicable asset by parsing text in documents and matching values to te value list.


## Dependencies

1. Python 3.7
2. Java 8
3. Microsoft Visual Studio BuildTools

## Running TaggerBot

1.	Install all Python packages in requirements.txt  
> `pip install –r requirements.txt`
2.	Download tika-server-1.22.jar jar file https://tika.apache.org/download.html
3.	Install Tesseract https://github.com/UB-Mannheim/tesseract/wiki
4.	Install Poppler for Windows and add bin/ folder to PATH environment
5.	Change directory to the directory where TaggerBot is stored
> `cd my_directory`
6.	Run TaggerBot’s tagger command, stating the inputFolder and outputFolder as parameters
> `python Taggerbot.tagger(inputFolder)`

> `inputFolder` is the folder which contains the files to be tagged



