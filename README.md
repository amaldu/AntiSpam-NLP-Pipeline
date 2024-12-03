# Spam Message Detector Model 



# Index of contents

1. [Objective](#Objective)
2. [About the data](#About-the-data)
3. [Methods used](#Methods-used)
4. [Contribución](#contribución)
5. [Licencia](#licencia)


# Objective

This project presents a step-by-step guide to building an efficient e-mail spam classification model.
By the end of this project, you'll have a powerful tool to help you filter out unwanted e-mails and ensure that your inbox is not filled with unnecessary content.

![spam classification](images/intro.png)


# About the data
The dataset used for this project can be found [here](https://www.kaggle.com/datasets/mfaisalqureshi/spam-email) which consists of 5,574 messages with the following columns: 

1. ***Category*** column with the following labels:

- HAM: real e-mails we want the filter to land our inbox
- SPAM: spam/scam em-amils that we want to send directly to the spam folder

2. ***Message*** column with a list of messages without any type of format

## Methods used

### [EDA](https://github.com/AMaldu/spam_detector/blob/main/notebooks/preprocessing.ipynb)
1. Analysis of basic information about the dataset.
2. Change data types for more memory efficiency and data integrity.
3. Removal of duplicates.
4. Basic viz of features.

### Cleaning: same notebook as before + [Special Replacements Analysis notebook](https://github.com/AMaldu/spam_detector/blob/main/notebooks/special_chars_analysis.ipynb)
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


### Base-model: BOW + Multinomial Naive Bayes 

**What is BOW and why using it?**

***BOW*** is a text representation technique that transforms text data into a set of features. Here's how it works:

1. Tokenization: The text is split into individual words or tokens
2. Vocabulary Creation: All unique words (or tokens) from the entire text corpus are collected to form a vocabulary
3. Vector Representation: Each document is represented as a vector where each element corresponds to the presence or frequency of a word from the vocabulary.

***BOW*** is an easy approach to the ham-spam problem that I considered useful for the following reasons:

1. BOW is simple to implement and understand
2. The semantic meaning of the e-mails is not that important because we are focusing on capturing keywords or phrases like  "free", "money", "offer", "limited time", "winner" and their frequencies can be indicative of spam or ham.
3. This is a vectorizer for the purpose of creating a base model. Let's keep it simple :)

**What is Multinomial Naive Bayes and why using it?**

The ***Multinomial Naive Bayes (MNB) model*** is a variant of the Naive Bayes algorithm that is particularly suited for classification tasks where the features are counts or frequencies of words in text data. It is called "multinomial" because it assumes that the features (typically word counts) follow a multinomial distribution like ours where the dependent variables are represented by the frequency of each word in the text data.

Naive Bayes classifiers are based on Bayes' Theorem and assume that the features (in this case, the words in a text document) are conditionally independent given the class. While this assumption is often unrealistic in practice, it simplifies the computation, making Naive Bayes a very efficient algorithm.

The Multinomial Naive Bayes is particularly effective for problems where the features are word counts or frequency counts of events, such as in text classification problems.

And why Multinomial?

Well, since I'm going to use a non-binary BOW and this is a first approach I will leave other Naive Bayes algorithms for later. What could I use? 

- Binary BOW + Bernoulli Naive Bayes, since it's suitable for working in the presence (1) or absence (0) or a word.

- If my dataset is very imbalaced I can use BOW + Complement Naive Bayes.




## Technologies

- Docker

## Installation


build docker container 

docker build -t mlflow .

run docker container

docker run -p 5000:5000 -v $(pwd)/mlflow_artifacts:/mlflow/artifacts mlflow


## Metrics used

Our the dataset is very imbalanced with a 13.41% of SPAM so it's important to choose the right metrics for the evaluation of the model. Those metrics have to provide a clear vision of the model performance in the minority class.

**Precision**

It measures the proportion of true positives (spam correctly classified) among all instances predicted as spam. A good precision for the spam class indicates that most e-mails classified as spam by the model are actually spam.

**Recall**

It measures the model's ability to capture all true positives (all spam instances). A good recall for the spam class indicates that the model is identifying most of the spam emails.

**F1-score**

It is the harmonic mean of precision and recall, meaning it considers both the model's ability to correctly identify positives (spam) and avoid false positives (ham classified as spam). We have to make sure the F1-score is high.

**Confusion Matrix**

The confusion matrix shows the number of true positives, false positives, true negatives, and false negatives.

<p align="center">
  <img src="images/Confusion-matrix-Precision-Recall-Accuracy-and-F1-score.jpg" width="500"/>
</p>


**ROC-AUC**

AUC measures the model’s ability to distinguish between classes. A ROC curve compares the true positive rate (recall) with the false positive rate, providing insight into how the model performs as decision thresholds change. In imbalanced datasets, a higher AUC means the model is better at separating the classes, even if the minority class is much less frequent. The AUC should be greater than 0.7 to be considered good.

**Precision-Recall AUC**

In imbalanced datasets like ours, the Precision-Recall curve is more informative than the ROC curve. This is because the ROC curve can be overly optimistic in imbalanced datasets, while the Precision-Recall curve is more useful in evaluating performance on the minority class (spam). The Precision-Recall AUC will tell us how well the model balances precision and recall for the minority class (spam).

**Balanced Accuracy**

Balanced accuracy is the average of the recall for both classes. It is useful in imbalanced settings because it gives equal weight to both classes. A balanced accuracy above 0.5 is considered good however, the closer it is to 1, the better the model handles both classes.


**ROC-AUC Curve**

The ROC curve plots the true positive rate (recall) against the false positive rate (1 - specificity) at different classification thresholds. AUC (Area Under the Curve) measures the overall ability of the model to distinguish between classes. It's important to minimize false negatives for spam.

#### Why not other metrics?

**Accuracy**

It measures the proportion of correct predictions (TPs & TNs) over all predictions. In an imbalanced dataset like ours where one class dominates (ham), a model can achieve high accuracy by mostly predicting the majority class. However, this doesn't necessarily mean the model is performing well.

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

