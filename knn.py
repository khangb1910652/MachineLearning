import numpy as np
from math import sqrt
import re

trainPath = input("Input path train file:")
trainf = open(trainPath, "r")
testPath = input("Input path test file:")
testf = open(testPath, "r")
k = input("Input number of nearest neighbors:")
dataTrain = np.array(list(trainf))# covert to matrix
trainf.close()
dataTest = np.array(list(testf))# covert to matrix
testf.close()
train = []
test = []
for item in dataTrain :
    train.append(item[:-1].split(" "))
for item in dataTest :
    test.append(item[:-1].split(" "))

def EuclideanDistance(row1, row2):
    distance = 0
    for i in range(len(row1)-1):
        distance += (float(row1[i]) - float(row2[i]))**2
    return sqrt(distance)
# print(data[1],data[2])
# print(EuclideanDistance(data[1],data[2]))

def getNeighbors(train, testRow, num):
    distance = list() # []
    data = []
    for i in train:
        dist = EuclideanDistance(testRow, i)
        distance.append(dist)
        data.append(i)
    distance = np.array(distance)
    data = np.array(data)
    index_dist = distance.argsort()
    data = data[index_dist]
    neighbors = data[:num]
    return neighbors
# print (test[0])
# print(getNeighbors(train,test[0],int(k)))

def predictClassification(train, testRow, num):
    neighbors = getNeighbors(train, testRow, num)
    classes = []
    for i in neighbors:
        classes.append(i[-1])
    predict = max(classes, key= classes.count)
    return predict
# print("We expected {}, Got {}".format(test[-1], predictClassification(train, test[0], int(k))))

def confusionMatrix(true, predict):
    classes = np.unique(true)
    matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
           matrix[i, j] = np.sum((np.array(true) == np.array(classes[i])) & (np.array(predict) == np.array(classes[j])))
    return matrix

def evaluate(true, predict):
    correct = 0
    for i in range(len(true)):
        if true[i] == predict[i]:
            correct += 1
    accuracy = correct/len(true)
    return accuracy

predict = []
true = []
trueTemp = []
for item in test:
    trueTemp.append(item[-1:])
    temp = predictClassification(train, item, int(k))
    predict.append(temp)
for item in trueTemp:
    true.append(item[0])
# print (predict)
# print (true)
print ("Confusion matrix: \n" , confusionMatrix(true, predict))
print ("Accuracy: " , evaluate(true, predict))
