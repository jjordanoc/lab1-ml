import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.use('macosx')

df = pd.read_csv("insurance.csv")
df.plot.scatter(x="bmi", y="charges")
df.plot.scatter(x="age", y="charges")


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(df["age"], df["bmi"], df["charges"], marker='o')


ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Charges')
ax.legend()


X = df[["age", "bmi"]].to_numpy()
rows, cols = X.shape
ones = np.ones((rows, 1), dtype=X.dtype)
X = np.append(X, ones, axis=1)
print(X, X.shape)

y = df["charges"].to_numpy()

XT = np.transpose(X)

b = np.matmul(np.linalg.inv(np.matmul(XT, X)), np.matmul(XT, y))

# Define the range of x and y values
xx1 = np.linspace(-10, 10, 100)
xx2 = np.linspace(-10, 10, 100)

Z = b[0] * xx1 + b[1] * xx2 + b[2]

ax.plot_surface(xx1, xx2, Z, color='b', alpha=0.5)

plt.show()

# X, Y = np.meshgrid(x, y)

print(b)