import numpy as np
import pandas as pd
from numpy.linalg import inv

# Read data from CSV files
Y = np.matrix(pd.read_csv('Y.csv'))
Z = np.matrix(pd.read_csv('Z.csv'))
TBV = np.matrix(pd.read_csv('TBV.csv'))
X = np.ones((Z.shape[0], 1))

# Additional setup
rX = X.shape[1]
p = Z.shape[1]
n = Y.shape[0]
k = Y.shape[1]

# Calculate InvXpX
InvXpX = inv(X.T @ X)

# Initialize variables
TrZSZ = 0
for i in range(p):
    TrZSZ += Z[:, i].T @ (Z[:, i] - X @ InvXpX @ X.T @ Z[:, i])

ZpZ = np.zeros(p)
for i in range(p):
    ZpZ[i] = Z[:, i].T @ Z[:, i]

XpX = np.zeros(rX)
for i in range(rX):
    XpX[i] = X[:, i].T @ X[:, i]

B = InvXpX @ X.T @ Y
SY = Y - X @ B
U = np.zeros((Z.shape[1], k))

Vy = np.diag(SY.T @ SY) / (n - rX)
Ve = Vy * 0.5
Vg = Ve / TrZSZ
Vg = np.diag(Ve) / TrZSZ
tilde = Z.T @ SY

convergence = 1
E = SY
iter = 0
maxit = 100

while (convergence > 1e-6) and (iter < maxit):
    # Store current Beta
   U_base = U * 1.0
   iter = iter + 1
   iVe = 1.0 / Ve
   iVg = inv(Vg)
   # Gauss-Seidel for random effects
   order = np.random.permutation(p)
   for i in order:
     U0 = U[i, :].copy()
     RHS = iVe * np.array(Z[:, i].T @ E + ZpZ[i] * U0)
     LHS = np.diag(iVe * ZpZ[i]) + iVg
     U[i, :] = np.linalg.solve(LHS, RHS.T).T
     E = E - Z[:, i] * (U[i, :] - U0)
   # Update fixed effects
   B_tmp = InvXpX @ X.T @ E
   B = B + B_tmp
   E = E - X @ B_tmp
   # Update VC
   Ve = np.diag(SY.T @ E) / (n - rX)
   Vg = tilde.T @ U / TrZSZ
   Vg = 0.5 * (Vg + Vg.T)
   # Update convergence
   convergence = np.max(np.sum((U - U_base) ** 2, axis=0))
   # Convergence
   print('iter', iter, '| conv', np.log10(convergence), '\n')




GEBV = Z @ U

Accuracy = np.diag(np.corrcoef(GEBV.T, TBV.T)[:k, k:])

print(np.mean(Accuracy))
