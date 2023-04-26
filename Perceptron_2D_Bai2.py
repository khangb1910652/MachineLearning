import numpy as np

def load_data(filename):
    X = []
    y = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data = line.strip().split(',')
            X.append([float(x) for x in data[:-1]])
            y.append(float(data[-1]))
    return np.array(X), np.array(y)

class Perceptron(object):
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
    # model training
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for epoch in range(self.epochs):
            y_pred = self.predict(X)
            accuracy = np.mean(y_pred == y)
            print("Epoch {}: Error = {:.2f}%".format(epoch, 100-accuracy*100))
            
            for i in range(X.shape[0]):
                if y[i] * (np.dot(X[i], self.w) + self.b) <= 0:
                    self.w += self.learning_rate * y[i] * X[i]
                    self.b += self.learning_rate * y[i]
    #prediction
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

def confusionMatrix(true, predict):
    classes = np.unique(true)
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
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

X_train, y_train = load_data(input("Input path train file:"))
X_test, y_test = load_data(input("Input path test file:"))
learning_rate = input("Input learning rate:")
epochs = input("Input number of epochs:")

model = Perceptron(float(learning_rate), int(epochs))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print ("Confusion matrix: \n" , confusionMatrix(y_test, y_pred))
print ("Accuracy: " , evaluate(y_test, y_pred))