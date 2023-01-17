import warnings
import math
import os
import pandas as pd
import numpy as np
import string
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.linear_model import SGDClassifier
import sys
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")


def updateBERSGDC(fileLocation):
    classLabel = ['0', '1']
    spamClass = classLabel[1]
    hamClass = classLabel[0]

    allTrainData, allTestData = getData(fileLocation, hamClass, spamClass)

    trainingA, trainingDiffWords = filterData(allTrainData.loc[:, allTrainData.columns != 'Class'])
    trainingDiffWords = list(trainingDiffWords)
    trainingA = goThroughData(trainingA, trainingDiffWords)

    testingA, testDiffWords = filterData(allTestData.loc[:, allTestData.columns != 'Class'])
    testingA = goThroughData(testingA, trainingDiffWords)

    trainingB = pd.DataFrame(allTrainData['Class'])
    testingB = pd.DataFrame(allTestData['Class'])

    return sgdclassifier(trainingA, trainingB, testingA, testingB)


def getData(thePath, hamClass, spamClass):
    spamTrainingData = getFromFile(thePath + '\\train\\spam', spamClass)
    hamTrainingData = getFromFile(thePath + '\\train\\ham', hamClass)
    allTrainData = hamTrainingData + spamTrainingData
    spamTestingData = getFromFile(thePath + '\\test\\spam', spamClass)
    hamTestingData = getFromFile(thePath + '\\test\\ham', hamClass)

    allTestData = hamTestingData + spamTestingData

    theTrainingDataFrame = pd.DataFrame(allTrainData, columns=['Documents', 'Class'])
    theTestingDataFrame = pd.DataFrame(allTestData, columns=['Documents', 'Class'])

    return theTrainingDataFrame, theTestingDataFrame


def getFromFile(dirName, className):
    allMessages = list()
    for filename in os.listdir(dirName):
        with open(os.path.join(dirName, filename), errors='replace', encoding='utf8') as f:
            fileContent = f.read().replace('\n', ' ')
            message = [str(fileContent), className]
            allMessages.append(message)
    random.seed(0)
    random.shuffle(allMessages)
    return allMessages


def filterData(data):
    x = string.printable
    waste = list(x)
    diffWords = list()
    diffWords.append('weight_zero')
    sw = stopwords.words('english')
    wordListFilter = []
    for index, row in data.iterrows():
        WordsList = row['Documents'].split()
        for token in WordsList:
            if token not in waste:
                wordListFilter.append(token)
        WordsList = wordListFilter
        wordListFilter = []
        for token in WordsList:
            if token.isalnum():
                wordListFilter.append(token)
        WordsList = wordListFilter
        wordListFilter = []
        for token in WordsList:
            if token not in sw:
                wordListFilter.append(token)
        WordsList = wordListFilter
        wordListFilter = []
        diffWords.extend(WordsList)
        data.loc[index, 'Documents'] = " ".join(WordsList)

    return data, set(diffWords)


def goThroughData(dataset, diffWords):
    lengthOfDiffWords = len(diffWords)
    setOfDataProcessed = pd.DataFrame(columns=diffWords)
    for idx, row in dataset.iterrows():
        data = [0] * lengthOfDiffWords
        for i in range(lengthOfDiffWords):
            if diffWords[i] == 'weight_zero':
                data[i] = 1
            else:
                if diffWords[i] in row['Documents']:
                    data[i] = 1
                else:
                    data[i] = 0
        setOfDataProcessed.loc[idx] = data
    return setOfDataProcessed.apply(pd.to_numeric)


def sgdclassifier(trainA, trainB, testA, testB):
    SGDC = SGDClassifier(random_state=0, loss='log', penalty='l2', class_weight='balanced', max_iter=1000)
    pGrid = {'alpha': [0.01, 0.1, 0.3]}

    SGDCGrid = GridSearchCV(estimator=SGDC, param_grid=pGrid, n_jobs=-1, scoring='roc_auc')
    SGDCGrid.fit(trainA, np.array(trainB))
    predictionClass = SGDCGrid.predict(testA)
    accuracy, precision, recall, F1 = evalPerfMetrics(testB, predictionClass)

    return accuracy, precision, recall, F1


def evalPerfMetrics(classY, classPrediction):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    i = 0
    for idx, row in classY.iterrows():
        if classPrediction[i] == row['Class']:
            if '1' == row['Class']:
                TN = TN + 1
            else:
                TP = TP + 1
        else:
            if '1' == row['Class']:
                FN = FN + 1
            else:
                FP = FP + 1
        i = i + 1
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * ((precision * recall) / (precision + recall))
    return accuracy, precision, recall, F1


if __name__ == '__main__':
    # print("Enter the number you would like to select.")
    # print("1. Multinomial Naive Bayes algorithm - uses Bag of words model")
    # print("2. Discrete Naive Bayes algorithm - uses Bernoulli model")
    # print("3. MCAP Logistic Regression algorithm with L2 regularization")
    # print("4. SGDClassifier")
    # typeOfAlgo = input("Your Input : ")
    #
    # print("Choose a file:")
    # print("1. enron1")
    # print("2. enron4")
    # print("3. hw1")
    # dataOption = input("Your Input: ")
    # if dataOption == 1:
    #     dataType = "./enron1"
    # elif dataOption == 2:
    #     dataType = "./enron4"
    # else:
    #     dataType = "./hw1"
    #
    # path = dataType
    #
    # if typeOfAlgo == 1:
    #     print("here")
    argList = sys.argv

    path = str(argList[1])
    accuracy, precision, recall, F1 = updateBERSGDC(path)
    print("Scores for SGDClassifier-Bern ")
    print("Accuracy  :", accuracy * 100)
    print("Precision :", precision * 100)
    print("Recall    :", recall * 100)
    print("F1        :", F1 * 100)