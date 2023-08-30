import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor
from random import randint

# mpl.use('macosx')

df = pd.read_csv("insurance.csv")


"""
A: Matrix of inputs [x1 x2; ...]
b: Vector of coefficients [c1, c2]
y: Vector of outputs
"""

# Calculate the coefficients assuming a zero-th point intersection (no bias)
A = df[["age", "bmi"]].to_numpy()
AT = np.transpose(A)
y = df["charges"].to_numpy()
b = np.matmul(np.linalg.inv(np.matmul(AT, A)), np.matmul(AT, y))
print("="*10,"Calculating Beta (no bias)","="*10)
print(f"The first coefficient assuming a zero-th point intersection is: {b[0]}")
print(f"The second coefficient assuming a zero-th point intersection is: {b[1]}")
print(f"Regressor plane: Y = {round(b[0], 2)}X1 + {round(b[1], 2)}X2")
print()



# Calculate the mean square error
n: int = y.shape[0]
e: np.array = y - np.matmul(A,b)
MSE: float = (1/n) * np.matmul(np.transpose(e), e)
print("="*10,"Calculating MSE (no bias)","="*10)
print("The mean squeare error is: ", round(MSE, 2))
print()

# Bootstrapping without bias
xx1_acumulate: float = 0
xx2_acumulate: float = 0

n_iter = 100
for i in range(n_iter): 
    row_removed: int = randint(1,len(df)-1)
    tmp_df = df.drop(df.index[row_removed])
    tmp_A = tmp_df[["age", "bmi"]].to_numpy()
    tmp_AT = np.transpose(tmp_A)
    tmp_y = tmp_df["charges"].to_numpy()
    tmp_b = np.matmul(np.linalg.inv(np.matmul(tmp_AT, tmp_A)), np.matmul(tmp_AT, tmp_y))
    xx1_acumulate += tmp_b[0]
    xx2_acumulate += tmp_b[1]

xx1_acumulate /= n_iter
xx2_acumulate /= n_iter
print("="*10,"Bootstrapping (no bias)","="*10)
print(f"The first coefficient doing bootstrapping without bias is: {xx1_acumulate}")
print(f"The second coefficient doing bootstrapping without bias is: {xx2_acumulate}")
print(f"Regressor plane: Y = {round(xx1_acumulate, 2)}X1 + {round(xx2_acumulate, 2)}X2")
print()

# Plot the raw data and a superimposing plane on the data that passes through the origin
min_age, max_age = floor(min(df["age"])), ceil(max(df["age"]))
min_bmi, max_bmi = floor(min(df["bmi"])), ceil(max(df["bmi"]))
xx1, xx2 = np.meshgrid(range(min_age, max_age + 1), range(min_bmi, max_bmi + 1))
Z = b[0] * xx1 + b[1] * xx2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Charges')
ax.plot_surface(xx1, xx2, Z, color='r', alpha=0.5)
ax.scatter(df["age"], df["bmi"], df["charges"], marker='o', color='b')
plt.show()

"""
B: Matrix of inputs [x1 x2 1; ...]
b: Vector of coefficients [c1, c2, bias]
y: Vector of outputs
"""

# Calculate the coefficients not assuming a zero-th point intersection (bias included)
B = df[["age", "bmi"]].to_numpy()
rows, cols = B.shape
ones = np.ones((rows, 1), dtype=B.dtype)
B = np.append(B, ones, axis=1)
y = df["charges"].to_numpy()
BT = np.transpose(B)
b = np.matmul(np.linalg.inv(np.matmul(BT, B)), np.matmul(BT, y))
print("="*10,"Calculating Beta (bias included)","="*10)
print(f"The first coefficient not assuming a zero-th point intersection is: {b[0]}")
print(f"The second coefficient not assuming a zero-th point intersection is: {b[1]}")
print(f"The bias term not assuming a zero-th point intersection is: {b[2]}")
print(f"Regressor plane: Y = {round(b[0], 2)}X1 + {round(b[1], 2)}X2 + {round(b[2],2)}")
print()

# Calculate the mean square error
n: int = y.shape[0]
e: np.array = y - np.matmul(B,b)
MSE: float = (1/n) * np.matmul(np.transpose(e), e)
print("="*10,"Calculating MSE (bias included)","="*10)
print("The mean squeare error is: ", round(MSE, 2))
print()

# Bootstrapping with bias
xx1_acumulate: float = 0
xx2_acumulate: float = 0
bias_acumulate: float = 0

n_iter = 100
for i in range(n_iter): 
    row_removed: int = randint(1,len(df)-1)
    tmp_df = df.drop(df.index[row_removed])
    tmp_B = tmp_df[["age", "bmi"]].to_numpy()
    rows, cols = tmp_B.shape
    tmp_ones = np.ones((rows,1), dtype=tmp_B.dtype)
    tmp_B = np.append(tmp_B, tmp_ones, axis=1)
    tmp_BT = np.transpose(tmp_B)
    tmp_y = tmp_df["charges"].to_numpy()
    tmp_b = np.matmul(np.linalg.inv(np.matmul(tmp_BT, tmp_B)), np.matmul(tmp_BT, tmp_y))
    xx1_acumulate += tmp_b[0]
    xx2_acumulate += tmp_b[1]
    bias_acumulate += tmp_b[2]

xx1_acumulate /= n_iter
xx2_acumulate /= n_iter
bias_acumulate /= n_iter
print("="*10,"Bootstrapping (bias included)","="*10)
print(f"The first coefficient doing bootstrapping with bias is: {xx1_acumulate}")
print(f"The second coefficient doing bootstrapping with bias is: {xx2_acumulate}")
print(f"The bias term not assuming a zero-th point intersection doing bootstrapping is: {bias_acumulate}")
print(f"Regressor plane: Y = {round(xx1_acumulate, 2)}X1 + {round(xx2_acumulate, 2)}X2 + {round(bias_acumulate,2)}")
print()

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

# # Predict a point using a plane ecuacion 
# def get_prediction_by_plane(beta2, beta1, bias, x2, x1):
#     return beta2*x2 + beta1*x1 + bias

# # Calculate the Mean Square Error
# def MSE(Yp: np.ndarray, Yr: np.ndarray):
#     n: int = Yr.shape[0]
#     return (1/n)*sum(pow(Yp-Yr,2))

# def MSE2(y: np.array, X: np.ndarray, b: np.array):
#     n: int = y.shape[0]
#     return 

# # Prediction of the N point's
# N = y.shape[0]
# Yp = np.array([ get_prediction_by_plane(b[0], b[1], b[2], B[i][1], B[i][0]) for i in range(N)])

# mse = MSE2(y, B, b)
