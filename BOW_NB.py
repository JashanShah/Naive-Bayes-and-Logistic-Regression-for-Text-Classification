import os
import pandas as pd
import math
import string
import sys
from nltk.corpus import stopwords




def updateMNNB(fileLocation):
    classLabel = ['0', '1']
    spamClass, hamClass = '1', '0'

    TrainingData, allTestData = extractData(fileLocation, hamClass, spamClass)
    # diffWords in vour Voacb
    #print(hamClass)
    TrainingData, diffWords = filterData(TrainingData)
    trainingA = pd.DataFrame(TrainingData['Documents'])
    trainingB = pd.DataFrame(TrainingData['Class'])
    diffWords = list(diffWords)
    trainingA = goThroughData(trainingA, diffWords)
    testingA, words = filterData(allTestData)
    testingA = pd.DataFrame(testingA, columns=['Documents'])
    testingY = pd.DataFrame(allTestData['Class'])

    trainingA['Class'] = trainingB['Class'].values
    testingA['Class'] = testingY['Class'].values

    bowTrainingData, bowTestingData = trainingA, testingA

    return multinomialNB(bowTrainingData, bowTestingData, classLabel)

def extractData(thePath, hamClass, spamClass):
    spamTrainingData = getFromFile(thePath + '\\train\\spam', spamClass)
    hamTrainingData = getFromFile(thePath + '\\train\\ham', hamClass)
    allTrainData = hamTrainingData + spamTrainingData
    spamTestingData = getFromFile(thePath + '\\test\\spam', spamClass)
    hamTestingData = getFromFile(thePath + '\\test\\ham', hamClass)
    #print('-------------heree------')
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
    return allMessages





def filterData(data):
    waste = list(string.printable)
    diffWords = list()
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
        newText = " ".join(WordsList)
        data.loc[index, 'Documents'] = newText

    return data, set(diffWords)


def goThroughData(dataset, diffWords):
    lengthOfDiffWords = len(diffWords)
    setOfDataProcessed = pd.DataFrame(columns=diffWords)
    for idx, row in dataset.iterrows():
        data = [0] * lengthOfDiffWords
        for i in range(lengthOfDiffWords):
            if diffWords[i] in row['Documents']:
                data[i] = row['Documents'].count(diffWords[i])
        setOfDataProcessed.loc[idx] = data
    return setOfDataProcessed.apply(pd.to_numeric)


def multinomialNB(trainingDataBOW, testingDataBOW, labelOfClass):
    v, prior, condprob = trainingMultinomialNB(trainingDataBOW, labelOfClass)
    classPrediction = testMultinomialNB(testingDataBOW, trainingDataBOW, v, labelOfClass, prior, condprob)
    accuracy, precision, recall, F1 = evalPerfMetrics(classPrediction, testingDataBOW)
    return accuracy, precision, recall, F1


def trainingMultinomialNB(data, classLabel):
    conditionalProb = dict()
    prior = dict()
    amt = data.shape[0]
    vocab = data.iloc[:, :-1].columns
    countOfVocab = len(vocab)
    for c in classLabel:
        tct = list()
        nc = (len(data[(data['Class'] == c)]))
        prior[c] = math.log(nc / amt, 2)

        for t in vocab:
            tct.append(data.loc[data['Class'] == c, t].sum())
        sumOfTct = sum(tct)
        i = 0
        for t in vocab:
            if t in conditionalProb:
                conditionalProb[t][c] = math.log((tct[i] + 1) / (countOfVocab + sumOfTct), 2)
            else:
                conditionalProb[t] = {c: (math.log((tct[i] + 1) / (countOfVocab + sumOfTct), 2))}
            i = i + 1
    return vocab, prior, conditionalProb


def testMultinomialNB(testingData, trainingData, vocab, classLabel, prior, conditionalProb ):
    spamLabel = str(classLabel[1])
    hamLabel = str(classLabel[0])
    predictions = list()
    spamData = trainingData[trainingData['Class'] == spamLabel]
    hamData = trainingData[trainingData['Class'] == hamLabel]
    wordCountSpam = spamData.iloc[:,:-1].sum(axis=1).sum()
    wordCountHam = hamData.iloc[:,:-1].sum(axis=1).sum()
    totalTrainingSpam = trainingData.shape[1] - 1 + wordCountSpam
    totalTrainingHam = trainingData.shape[1] - 1 + wordCountHam
    calcPriorWSpam = -1 * math.log(totalTrainingSpam, 2)
    calcPriorWHam = -1 * math.log(totalTrainingHam, 2)
    spamProb = prior[spamLabel]
    hamProb = prior[hamLabel]
    for index, row in testingData.iterrows():
        diffWords = set(row['Documents'].split())
        spamEmailProb = spamProb
        hamEmailProb = hamProb
        for w in diffWords:
            if w not in vocab:
                prWS = calcPriorWSpam
                prWH = calcPriorWHam
            else:
                prWS = conditionalProb[w][spamLabel]
                prWH = conditionalProb[w][hamLabel]
            spamEmailProb = prWS + spamEmailProb
            hamEmailProb = prWH + hamEmailProb
        if hamEmailProb > spamEmailProb:
            predictions.append(hamLabel)
        else:
            predictions.append(spamLabel)
    return pd.DataFrame(predictions, columns=['Class'])


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
    #path = os.getcwd()
    path = str(arg_list[1])
    accuracy, precision, recall, F1 = updateMNNB(path)
    print("Scores for Multinomial Naive Bayes-Bag Of Words ")
    print("Accuracy  :", accuracy * 100)
    print("Precision :", precision * 100)
    print("Recall    :", recall * 100)
    print("F1        :", F1 * 100)

