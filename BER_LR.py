import os
import warnings

import pandas as pd
import numpy as np
import math
import random
import string
import sys

from nltk.corpus import stopwords

warnings.filterwarnings("ignore")


def updateLR(fileLocation):
    classLabel = ['0', '1']
    hamClass = classLabel[0]
    spamClass = classLabel[1]
    allTrainData, allTestData, validData, totalTrainData = getData(fileLocation, hamClass, spamClass)

    trainingA, trainingDiffWords = filterData(allTrainData.loc[:, allTrainData.columns != 'Class'])
    trainingDiffWords = list(trainingDiffWords)
    validA, validDiffWords = filterData(validData.loc[:, validData.columns != 'Class'])

    trainingB = pd.DataFrame(allTrainData['Class'])
    testingY = pd.DataFrame(allTestData['Class'])
    validB = pd.DataFrame(validData['Class'])
    totalTrainingDataY = pd.DataFrame(totalTrainData['Class'])

    trainingA = goThroughData(trainingA, trainingDiffWords)
    validA = goThroughData(validA, trainingDiffWords)

    finalLambda = chooseLambda(trainingA, trainingB, validA, validB)
    print("The Final lambda is ", finalLambda)

    totalTrainingDataX, totalTrainingDiffWords = filterData(totalTrainData.loc[:, totalTrainData.columns != 'Class'])
    totalTrainingDiffWords = list(totalTrainingDiffWords)
    totalTrainingDataX = goThroughData(totalTrainingDataX, totalTrainingDiffWords)
    weights = np.zeros(totalTrainingDataX.shape[1])
    weights = logisticRegression(totalTrainingDataX, totalTrainingDataY, weights, finalLambda)

    testingX, testingUniqueWords = filterData(allTestData.loc[:, allTestData.columns != 'Class'])
    testingX = goThroughData(testingX, totalTrainingDiffWords)
    allTestData = testingX.values
    accuracy, precision, recall, F1 = evalPerfMetrics(allTestData, weights, testingY)

    return accuracy, precision, recall, F1


def getData(fileLocation, hamClass, spamClass):
    spamTrainingData = getFromFile(fileLocation + '/train/spam', spamClass)
    hamTrainingData = getFromFile(fileLocation + '/train/ham', hamClass)
    allTrainData = hamTrainingData + spamTrainingData
    spamTestingData = getFromFile(fileLocation + '/test/spam', spamClass)
    hamTestingData = getFromFile(fileLocation + '/test/ham', hamClass)
    allTestData = hamTestingData + spamTestingData
    hamTrainingData, hamValidData = splitIntoTrainingValid(hamTrainingData)
    spamTrainingData, spamValidData = splitIntoTrainingValid(spamTrainingData)
    splitTrainingData = hamTrainingData + spamTrainingData
    splitValidData = hamValidData + spamValidData
    splitTrainingData = pd.DataFrame(splitTrainingData, columns=['Documents', 'Class'])
    allTestData = pd.DataFrame(allTestData, columns=['Documents', 'Class'])
    splitValidData = pd.DataFrame(splitValidData, columns=['Documents', 'Class'])
    allTrainData = pd.DataFrame(allTrainData, columns=['Documents', 'Class'])
    return splitTrainingData, allTestData, splitValidData, allTrainData

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


def splitIntoTrainingValid(data):
    validationData = []
    theLength = len(data)
    allTrainData = []
    d = (0.7 * theLength)

    for i in range(theLength):
        if i < d:
            allTrainData.append(data[i])
        else:
            validationData.append(data[i])

    return allTrainData, validationData


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
    setOfDataProcessed = pd.DataFrame(columns=diffWords)
    lengthOfDiffWords = len(diffWords)
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


def chooseLambda(dataA, dataB, validDataA, validDataB):
    lambdaVals = [0.1, 0.3, 0.5, 0.7, 0.9]
    minAcc = 0
    finalLambda = lambdaVals[0]
    for lamVal in lambdaVals:
        weights = np.zeros(dataA.shape[1])
        for i in range(300):
            X = np.array(np.dot(dataA, weights), dtype=np.float32)
            S = sigmoid(X)
            C = np.array(dataB.apply(pd.to_numeric)).reshape(dataB.shape[0])
            g = np.dot(dataA.T, C - S)
            weights = weights + (0.01 * g) - (0.01 * lamVal * weights)
        validData = validDataA.values
        accuracy, precision, recall, F1 = evalPerfMetrics(validData, weights, validDataB)
        if minAcc <= accuracy:
            finalLambda = lamVal
            minAcc = accuracy
    return finalLambda


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def logisticRegression(trainingA, trainingB, weights, finalLambda):
    for i in range(1000):
        x = np.array(np.dot(trainingA, weights), dtype=np.float32)
        S = sigmoid(x)
        C = np.array(trainingB.apply(pd.to_numeric)).reshape(trainingB.shape[0])
        g = np.dot(trainingA.T, C - S)
        weights = weights + (0.01 * (g)) - (0.01 * finalLambda * weights)
    return weights


def evalPerfMetrics(data, weights, classB):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    predClass = []
    predB = np.array(np.dot(data, weights), dtype=np.float32)
    for i in predB:
        if i < 0:
            predClass.append('0')
        else:
            predClass.append('1')
    i = 0
    for idx, row in classB.iterrows():
        if predClass[i] == row['Class']:
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
    accuracy, precision, recall, F1 = updateLR(path)
    print("Scores for Logistic Regression-Bernoulli")
    print("Accuracy  :", accuracy * 100)
    print("Precision :", precision * 100)
    print("Recall    :", recall * 100)
    print("F1        :", F1 * 100)
