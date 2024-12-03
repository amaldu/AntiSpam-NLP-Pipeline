# Spam Message Detector Model 

## Objective

Thisproject presents a step-by-step guide to building an efficient e-mail spam classification model using the e-mail Spam Collection dataset. 

By the end of this project, you'll have a powerful tool to help you filter out unwanted e-mails and ensure that your inbox is not filled with unnecessary content.

![spam classification](images/intro.png)


## About the data
The dataset used for this project can be found [here](https://www.kaggle.com/datasets/mfaisalqureshi/spam-email) which consists of 5,574 messages with the following columns: 

1. "Category" column with the following labels:

    * HAM: real e-mails we want the filter to land our inbox
    * SPAM: spam/scam em-amils that we want to send directly to the spam folder

2. "Message" column with a list of messages without any type of format

## Methods Used

### [EDA](https://github.com/AMaldu/spam_detector/blob/main/notebooks/preprocessing.ipynb)
    1. Viz of basic information about the dataset.
    2. Change data types for more memory efficiency and data integrity.
    3. Removal of duplicates.
    4. Basic viz of features.

### Cleaning: same notebook as before + [Special replacements analysis](https://github.com/AMaldu/spam_detector/blob/main/notebooks/special_chars_analysis.ipynb)
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


### Base model: Multinomial Naive Bayes 

The Multinomial Naive Bayes (MNB) model is a variant of the Naive Bayes algorithm that is particularly suited for classification tasks where the features are counts or frequencies of words (or other types of features) in text data. It is called "multinomial" because it assumes that the features (typically word counts) follow a multinomial distribution.

Naive Bayes classifiers are based on Bayes' Theorem and assume that the features (in this case, the words in a text document) are conditionally independent given the class. While this assumption is often unrealistic in practice, it simplifies the computation, making Naive Bayes a very efficient algorithm.

The Multinomial Naive Bayes is particularly effective for problems where the features are word counts or frequency counts of events, such as in text classification problems.



## Technologies

- Docker

## Installation


build docker container 

docker build -t mlflow .

run docker container

docker run -p 5000:5000 -v $(pwd)/mlflow_artifacts:/mlflow/artifacts mlflow


## Metrics used

**Precision**: measures the accuracy of the positive predictions made by the model

**Recall**: measures the ability of the model to detect all the instances of the positive class

**F1-score**: combines precision and recall and it's useful when there is an imbalance between the classes (our case)


<p align="center">
  <img src="images/Confusion-matrix-Precision-Recall-Accuracy-and-F1-score.jpg" width="500"/>
</p>

**ROC-AUC Curve**: the ROC curve plots the true positive rate (recall) against the false positive rate (1 - specificity) at different classification thresholds. AUC (Area Under the Curve) measures the overall ability of the model to distinguish between classes.

#### Why not other metrics?

**Accuracy**: measures the proportion of correct predictions (TPs & TNs) over all predictions. In an imbalanced dataset like ours where one class dominates (ham), a model can achieve high accuracy by mostly predicting the majority class. However, this doesn't necessarily mean the model is performing well.

Choosing precision, recall and f1-score is better because they focus on the model's performance of the minority class too.

## Experimentation: 

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


## Conclusions

