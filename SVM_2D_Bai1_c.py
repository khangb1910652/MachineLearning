import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Dataset
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([-1,1,1,1])

# Using svm linear and c = 100
model = svm.SVC(kernel='linear', C=100)
model.fit(x, y)

# Draw the boundary line
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.show()
