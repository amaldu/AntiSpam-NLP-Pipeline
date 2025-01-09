# AntiSpam NLP Pipeline

# Index of contents
1. [Objective](#Objective)
2. [About the data](#About-the-data)
3. [Research](#Research) Theory behind Decision-Making Process 🚧
    - [EDA](#EDA)
    - [Preprocessing](#Preprocessing)
    - [Experimentation](#Experimentation) Theory behind Decision-Making Process 🚧
6. [Technologies](#Technologies) 🚧
7. [Installation](#Installation) 🚧


# Objective
This purpose of this project is to create an end-to-end Text Classifier that classifies spam e-mails based on p guide to build an e-mail spam classification model.

<p align="center">
  <img src="images/intro.png" width="500"/>
</p>




# About the data
This project is based on four datasets extracted from Kaggle

[Spam Email](https://www.kaggle.com/datasets/mfaisalqureshi/spam-email) 
 5,572 messages with the following columns: 

1. ***Category*** column with the following labels:

- HAM: real e-mails we want the filter to land our inbox
- SPAM: spam/scam em-amils that we want to send directly to the spam folder

2. ***Message*** column with a list of messages without any type of format

[Spam Email Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset) 
83446 messages with the following columns:

1. ***label*** column with the following labels:

- 0: real e-mails we want the filter to land our inbox
- 1: spam/scam em-amils that we want to send directly to the spam folder

2. ***text*** column with a list of messages without any type of format

[Email Classification (Ham-Spam)](https://www.kaggle.com/datasets/prishasawhney/email-classification-ham-spam) 
179 messages with the following columns:

1. ***label*** column with the following labels:

- ham: real e-mails we want the filter to land our inbox
- spam: spam/scam em-amils that we want to send directly to the spam folder

2. ***email*** column with a list of messages without any type of format

[Spam email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) 
5695 messages with the following columns:

1. ***spam*** column with the following labels:

- 0: real e-mails we want the filter to land our inbox
- 1: spam/scam em-amils that we want to send directly to the spam folder

2. ***text*** column with a list of messages without any type of format



# Research

## EDA
[Notebook](https://github.com/AMaldu/spam_detector/blob/main/research/eda.ipynb)
1. Basic overview of the dataset.

4. Basic viz of features.    

## Preprocessing

[Notebook](https://github.com/AMaldu/spam_detector/blob/main/research/preprocessing.ipynb)

1. Change of data types
2. Drop duplicates
3. Chars cleaning. Based on this [Notebook](https://github.com/AMaldu/spam_detector/blob/main/research/special_chars_analysis.ipynb)
    1. Replacement of special replacements  
    2. Replacement of emojis
    3. Conversion to lowecase
    4. Removal of HTML tags
    5. Removal of URLs
    6. Replacement of numbers with "number"
    7. Replacement of e-mail addresses with "emailaddr"
    8. Removal of punctutation
    9. Removal of Non-Alphabetic Characters
    10. Collapse of multiple whitespaces into single whitespace

4. Tokenization
5. Removal of stopwords
6. Lemmatization





# Experimentation
This section only contains results. If you want to take a look at the Decision-Making Process please take a look at the Experimentation [README file](https://github.com/AMaldu/spam_detector/blob/main/experiments/EXPERIMENTS_GUIDELINE.md) 
1. Base model BOW + Multinomial Naive Bayes


Classification Report (Test):
                 
                  precision    recall  f1-score   support

           0       0.99      0.97      0.98       453
           1       0.82      0.95      0.88        63

    accuracy                           0.97       516
    macro avg      0.91      0.96      0.93       516
    weighted avg   0.97      0.97      0.97       516

- Recall 0.95: the model is detecting a 95% of the class 1

- Precision 0.82: lower than we would like to because there is a significative number of False Positives

- F1-score 0.88: decent value but since recall is high and precission not that much there is room for improvement.
























# Technologies

- Docker

- aws 




# Installation


build docker container 

docker build -t mlflow .

run docker container

docker run -p 5000:5000 -v $(pwd)/mlflow_artifacts:/mlflow/artifacts mlflow




# Conclusions

