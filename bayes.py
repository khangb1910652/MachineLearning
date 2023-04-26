import numpy as np
from sklearn.naive_bayes import GaussianNB

trainPath = input("Input path train file:")
trainf = open(trainPath, "r")
testPath = input("Input path test file:")
testf = open(testPath, "r")
# k = input("Input number of nearest neighbors:")
dataTrain = np.array(list(trainf))# covert to matrix
trainf.close()
dataTest = np.array(list(testf))# covert to matrix
testf.close()
train = []
test = []

for item in dataTrain :
    train.append(item[:-1].split(","))
for item in dataTest :
    test.append(item[:-1].split(","))

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
truexTemp = []
trueyTemp = []
truex = []
truey = []
trainxTemp = []
trainyTemp = []
trainx = []
trainy = []
for item in test:
    truexTemp.append(item[:-1])
    trueyTemp.append(item[-1:])
for item in truexTemp:
    temp = []
    for i in item:
        temp.append(float(i))
    truex.append(temp)    
for item in trueyTemp:
    truey.append(int(item[0]))

for item in train:
    trainxTemp.append(item[:-1])
    trainyTemp.append(item[-1:])
for item in trainxTemp:
    temp = []
    for i in item:
        temp.append(float(i))
    trainx.append(temp)    
for item in trainyTemp:
    trainy.append(int(item[0]))

clf = GaussianNB()
clf.fit(np.array(trainx),np.array(trainy))
predict = clf.predict(truex)

# print (predict)
# print (truey)
print ("Confusion matrix: \n" , confusionMatrix(truey, predict))
print ("Accuracy: " , evaluate(truey, predict))
