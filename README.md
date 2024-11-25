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

1. Punctuation removal 
    1. Remove urls, @users and numbers
    2. Remove every character that is not alphanumeric (like #)
    3. Lower every letter
    3. Delete extra whitespaces





## Technologies

- Docker

## Installation


build docker container 

docker build -t mlflow .

run docker container

docker run -p 5000:5000 -v $(pwd)/mlflow_artifacts:/mlflow/artifacts mlflow


## Conclusions

