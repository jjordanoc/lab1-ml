import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor

mpl.use('macosx')

df = pd.read_csv("insurance.csv")

y = df["charges"].to_numpy()

# Calculate the coefficients assuming a zero-th point intersection (no bias)
A = df[["age", "bmi"]].to_numpy()
AT = np.transpose(A)
b = np.matmul(np.linalg.inv(np.matmul(AT, A)), np.matmul(AT, y))
print(f"The first coefficient assuming a zero-th point intersection is: {b[0]}")
print(f"The second coefficient assuming a zero-th point intersection is: {b[1]}")
print(f"Regressor plane: Y={round(b[0], 2)}X1+{round(b[1], 2)}X2")


# Calculate the bias coefficients not assuming a zero-th point intersection (bias included)
B = df[["age", "bmi"]].to_numpy()
rows, cols = B.shape
ones = np.ones((rows, 1), dtype=B.dtype)
B = np.append(B, ones, axis=1)
y = df["charges"].to_numpy()
BT = np.transpose(B)
b = np.matmul(np.linalg.inv(np.matmul(BT, B)), np.matmul(BT, y))
print(f"The first coefficient not assuming a zero-th point intersection is: {b[0]}")
print(f"The second coefficient not assuming a zero-th point intersection is: {b[1]}")
print(f"The bias term not assuming a zero-th point intersection is: {b[2]}")
print(f"Regressor plane: Y={round(b[0], 2)}X1+{round(b[1], 2)}X2+{round(b[2],2)}")


# Plot the raw data and a superimposing plane on the data that passes through the origin
min_age, max_age = floor(min(df["age"])), ceil(max(df["age"]))
min_bmi, max_bmi = floor(min(df["bmi"])), ceil(max(df["bmi"]))
xx1, xx2 = np.meshgrid(range(min_age, max_age + 1), range(min_bmi, max_bmi + 1))
Z = b[0] * xx1 + b[1] * xx2 + b[2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Charges')
ax.plot_surface(xx1, xx2, Z, color='r', alpha=0.5)
ax.scatter(df["age"], df["bmi"], df["charges"], marker='o', color='b')
plt.show()

# Plot the raw data and a superimposing plane on the data that does not pass through the origin
min_age, max_age = floor(min(df["age"])), ceil(max(df["age"]))
min_bmi, max_bmi = floor(min(df["bmi"])), ceil(max(df["bmi"]))
xx1, xx2 = np.meshgrid(range(min_age, max_age + 1), range(min_bmi, max_bmi + 1))
Z = b[0] * xx1 + b[1] * xx2 + b[2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Charges')
ax.plot_surface(xx1, xx2, Z, color='r', alpha=0.5)
ax.scatter(df["age"], df["bmi"], df["charges"], marker='o', color='b')
plt.show()
