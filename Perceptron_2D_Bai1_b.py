import matplotlib.pyplot as plt

plt.xlim(-1.5,2.5)
plt.ylim(-1.5,2.5)
# class +1
plt.scatter([0,1,1], [1,0,1])
# class -1
plt.scatter([0], [0])

# -0.1 + 0.4 * x1 + 0.9 * x2 = 0 with 2 point (-1,2.5), (1,-2)
plt.plot([-2,2.5],[1,-1], label="-0.1 + 0.4 * x1 + 0.9 * x2 = 0" , linewidth=0.5)
plt.legend(loc = "upper left")

plt.show()

