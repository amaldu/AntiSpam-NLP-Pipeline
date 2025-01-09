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

<p align="center">
  <img src="images/intro.png" width="500"/>
</p>

# Project Developer Roadmap

This project started long time ago as a data science project for my personal portfolio with the goal of gaining experience in working with NLP techniques for text classification but I decided to make it grow and create a full implemented project with Machine Learning, Deep Learning (LLMs) and MLOps methods.

# About the data
This project is based on four datasets extracted from Kaggle

[Spam Email](https://www.kaggle.com/datasets/mfaisalqureshi/spam-email) with 5,572 rows 
[Spam Email Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset) with 83446 rows 
[Email Classification (Ham-Spam)](https://www.kaggle.com/datasets/prishasawhney/email-classification-ham-spam) with 179 rows 
[Spam email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) with 5695 rows 

All datasets have a similar format consisting on two columns:

1. ***Message/text/email*** that contains the emails
2. ***Message/label/spam*** that contains ham/spam or 0/1 values

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

