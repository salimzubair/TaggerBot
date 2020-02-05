# TaggerBot

TaggerBot is a machine learning application that reads all documents in document repositories (e.g. Shared Drives, SharePoint, etc.) and applies Natural Language Processing (NLP) and Natural Language Understanding (NLU) techniques to provide summaries, metadata and structure to the files in the repositories. TaggerBot tags each document with indexes, descriptors and metadata to assist users in information retrieval.

TaggerBot was originally built to solve an enterprise problem my client (a large public sector company with 2 million customers) had with their files. My client, as well as other companies, had a problem of having more documents than they could read or make sense of. TaggerBot was built to automatically classify millions of documents. TaggerBot classifies whether a document is useful to keep or transitory, it also determines the following metadata fields:

- The subject of the document
- Whether or not the document contained sensitive data
- A summary of the document
- Top keywords in the document
- The project ID and project name the document relates to
- Geographic points related to the document
- Author of the document

TaggerBot determines all the above by parsing the text within the document and using machine learning algorithms and rule-based programming to determine the apprpriate metadata tags. Note that TaggerBot was trained specifically for this corporation and therefore, would need to be trained with new data for other companies and use cases.


## High Level TaggerBot Process




## Metadata

1. **Document Type**: TaggerBot uses a Multilayer Perceptron Classifier algorithm to determine the document type of the document.The document type is a signle label field with 11 classes.

2. **Subject Area**: TaggerBot uses a Multilayer Perceptron Classifier algorithm to determine the subject area of the document. The subject area field is a mult-label field with 8 classes.

3. **Contains Sensitive Data**: TaggerBot uses a Multilayer Perceptron Classifier algorithm to determine whether or not a document contains sensitive data. This is a binary class field.

4. **Document Summary**: Taggerbot uses an (blank) which is an extractive text summarization module to give a brief summary of the document.

5. **Keywords**: Taggerbot uses 
