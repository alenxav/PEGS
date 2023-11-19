setwd("C:/Users/rd7564.PHIBRED/Desktop/julia_pegs")

Y = as.matrix(read.csv('Y.csv'))
Z = as.matrix(read.csv('Z.csv'))
TBV = as.matrix(read.csv('TBV.csv'))
X = matrix(1,ncol = 1,nrow = nrow(Z))

###

rX = ncol(X)
p = ncol(Z)
n = nrow(Y)
k = ncol(Y)


InvXpX = solve( t(X) %*% X )
TrZSZ = 0
for(i in 1:p) TrZSZ = TrZSZ + t(Z[,i]) %*% (Z[,i] - X %*% InvXpX %*% t(X) %*% Z[,i])

ZpZ = rep(0,p)
for(i in 1:p) ZpZ[i] = t(Z[,i]) %*% Z[,i] 
XpX = rep(0,rX)
for(i in 1:rX) XpX[i] = t(X[,i]) %*% X[,i] 

B = InvXpX %*% t(X) %*% Y
SY = Y - X %*% B
U = matrix(0,nrow=ncol(Z),ncol=k)

Vy = diag( t(SY) %*% SY ) / (n - rX)
Ve = Vy*0.5
Vg = diag(Ve/c(TrZSZ))
tilde = t(Z) %*% SY

convergence = 1
E = SY
iter = 0
maxit = 100

while(convergence>(1e-8)&iter<maxit){
  
  # Store current Beta
  U_base = U
  iter = iter+1
  
  iVe = 1/Ve
  iVg = solve(Vg)

  # Gauss-Seidel for random effects
  order = sample(p)
  for(i in order){
    U0 = U[i,]
    RHS = iVe * t(Z[,i] %*% E + ZpZ[i]*U0)
    LHS = diag(iVe*ZpZ[i]) + iVg
    U[i,] = solve(LHS,RHS)
    E = E - Z[,i] %*% t(U[i,]-U0)
  }
  
  # Update fixed effects
  B_tmp = InvXpX %*% t(X) %*% E
  B = B+B_tmp
  E = E - X %*% B_tmp
  
  # Update VC
  Ve = diag(t(SY) %*% E) / (n-rX)
  Vg = t(tilde) %*% U / c(TrZSZ)
  Vg = 0.5 * ( Vg + t(Vg) ) 
  
  # Update convergence
  convergence = max( colSums( (U-U_base)^2 ) )
  
  # Convergence
  cat('iter',iter,'| conv', log10(convergence),'\n')
  
}

GEBV = Z %*% U

h2 = 1-Ve/Vy
Accuracy = diag(cor(GEBV,TBV))

print(mean(Accuracy))

