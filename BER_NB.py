import os
import pandas as pd
import numpy as np
import math
import string
import sys

from nltk.corpus import stopwords


def updateDiscreteNB(fileLocation):
    spamClass = '1'
    hamClass = '0'
    classLabel = ['0', '1']

    allTrainData, allTestData = getData(fileLocation, hamClass, spamClass)
    allTrainData, diffWords = filterData(allTrainData)
    diffWords = list(diffWords)

    trainingA = pd.DataFrame(allTrainData['Document'])
    trainingB = pd.DataFrame(allTrainData['Class'])
    trainingA = goThroughData(trainingA, diffWords)

    testingA, words = filterData(allTestData)
    testingA = pd.DataFrame(testingA, columns=['Document'])
    testingB = pd.DataFrame(allTestData['Class'])

    trainingA['Class'] = trainingB['Class'].values
    testingA['Class'] = testingB['Class'].values

    trainingDataBERN, testingDataBERN = trainingA, testingA
    return discreteNB(trainingDataBERN, testingDataBERN, classLabel)


def getData(fileLocation, hamClass, spamClass):
    SpamTrainData = getFromFile(fileLocation + '/train/spam', spamClass)
    HamTrainData = getFromFile(fileLocation + '/train/ham', hamClass)
    allTrainData = HamTrainData + SpamTrainData

    SpamTestData = getFromFile(fileLocation + '/test/spam', spamClass)
    HamTestData = getFromFile(fileLocation + '/test/ham', hamClass)
    allTestData = HamTestData + SpamTestData

    allTrainData = pd.DataFrame(allTrainData, columns=['Document', 'Class'])
    allTestData = pd.DataFrame(allTestData, columns=['Document', 'Class'])
    return allTrainData, allTestData


def getFromFile(dirName, className):
    allMessages = list()
    for filename in os.listdir(dirName):
        with open(os.path.join(dirName, filename), errors='replace', encoding='utf8') as f:
            fileContent = f.read().replace('\n', ' ')
            message = [str(fileContent), className]
            allMessages.append(message)
    return allMessages


def filterData(data):
    x = string.printable
    waste = list(x)
    diffWords = list()
    sw = stopwords.words('english')
    wordListFilter = []
    for index, row in data.iterrows():
        WordsList = row['Document'].split()
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
        data.loc[index, 'Document'] = " ".join(WordsList)
    return data, set(diffWords)


def goThroughData(dataset, diffWords):
    lengthOfDiffWords = len(diffWords)
    setOfDataProcessed = pd.DataFrame(columns=diffWords)
    for idx, row in dataset.iterrows():
        data = [0] * lengthOfDiffWords
        for i in range(lengthOfDiffWords):
            if diffWords[i] in row['Document']:
                data[i] = 1
            else:
                data[i] = 0
        setOfDataProcessed.loc[idx] = data
    return setOfDataProcessed.apply(pd.to_numeric)


def DiscreteNB(classLabel, vocab, prior, condProb, data, allTrainData):
    spamLabel = str(classLabel[1])
    hamLabel = str(classLabel[0])
    prediction = list()
    spamMailCount = (len(allTrainData[(allTrainData['Class'] == spamLabel)]))
    hamMailCount = (len(allTrainData[(allTrainData['Class'] == hamLabel)]))
    for index, row in data.iterrows():
        spamEmailProb = 0
        hamEmailProb = 0
        diffWords = set(row['Document'].split())
        for words in diffWords:
            if words in vocab:
                prWS = condProb[words][spamLabel]
                prWH = condProb[words][hamLabel]
                prSW = prWS
                prHW = prWH
            else:
                prWS = -math.log(2 + spamMailCount, 2)
                prWH = -math.log(2 + hamMailCount, 2)
                prSW = prWS
                prHW = prWS
            spamEmailProb = prSW + spamEmailProb
            hamEmailProb = prHW + hamEmailProb
        if hamEmailProb > spamEmailProb:
            prediction.append(hamLabel)
        else:
            prediction.append(spamLabel)
    return pd.DataFrame(prediction, columns=['Class'])


def trainDiscreteNB(data, classLabel):
    prior = dict()
    condProb = dict()
    amt = data.shape[0]
    vocab = data.iloc[:, :-1].columns


    for c in classLabel:
        nc = (len(data[(data['Class'] == c)]))
        prior[c] = math.log(nc/amt, 2)
        nct = list()
        for t in vocab:
            nct.append(len(data[(data['Class'] == c) & (data[t] == 1)]))
        i = 0
        for t in vocab:
            if t in condProb:
                condProb[t][c] = math.log((1 + nct[i])/ (2 + nc), 2)
            else:
                condProb[t] = {c: math.log((1 + nct[i]) / (2 + nc), 2)}
            i = i + 1
    return vocab, prior, condProb





def evalPerfMetrics(classPrediction, data):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in data.index:

        isTN = classPrediction['Class'][i] == str(1) and data['Class'][i] == str(1)
        isFP = classPrediction['Class'][i] == str(1) and data['Class'][i] == str(0)
        isFN = classPrediction['Class'][i] == str(0) and data['Class'][i] == str(1)

        if(isTN):
            TN = TN + 1

        elif (isFP):
            FP = FP + 1

        elif(isFN):
            FN = FN + 1

        else:
            TP = TP + 1
    print("TP        :", TP)
    print("TN        :", TN)
    print("FP        :", FP)
    print("FN        :", FN)
    print("*******************************************************")

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall  = TP / (TP + FN)
    F1 = 2 * ((precision * recall) / (precision + recall))

    return accuracy, precision, recall, F1


def discreteNB(trainingDataBERN, testingDataBERN, classLabel):
    v, prior, condprob = trainDiscreteNB(trainingDataBERN, classLabel)
    predClass = DiscreteNB(classLabel, v, prior, condprob, testingDataBERN, trainingDataBERN)
    accuracy, precision, recall, F1 = evalPerfMetrics(predClass, testingDataBERN)

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


    arg_list = sys.argv

    path = str(arg_list[1])
    accuracy, precision, recall, F1 = updateDiscreteNB(path)
    print("Scores for Discrete Naive Bayes-Bernoullli ")
    print("Accuracy  :", accuracy * 100)
    print("Precision :", precision * 100)
    print("Recall    :", recall * 100)
    print("F1        :", F1 * 100)

