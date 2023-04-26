import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Dataset
x = np.array([[0.204,0.834], [0.222,0.730], [0.298,0.822], [0.450,0.842], [0.412,0.732],
              [0.298,0.640], [0.588,0.298], [0.554,0.398], [0.670,0.466], [0.834,0.426],
              [0.724,0.368], [0.790,0.262], [0.824,0.338], [0.136,0.260], [0.146,0.374],
              [0.258,0.422], [0.292,0.282], [0.478,0.568], [0.654,0.776], [0.786,0.758],
              [0.690,0.628], [0.736,0.786], [0.574,0.742]])
y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])

# Using svm linear and c = 100
clf = svm.SVC(kernel='rbf', C=1000)
clf.fit(x, y)

# Draw the boundary line
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.show()
