# Set working directory
cd("C:/Users/rd7564.PHIBRED/Desktop/julia_pegs")

using DelimitedFiles
using LinearAlgebra
using Random
using Statistics

# Read data
Y = readdlm("Y.csv", ',', Float64)
Z = readdlm("Z.csv", ',', Float64)
TBV = readdlm("TBV.csv", ',', Float64)
n = size(Y, 1)
X = ones(Int, n, 1)

# Dimensions
rX = size(X, 2)
p = size(Z, 2)
k = size(Y, 2)

# Initialize matrices
InvXpX = inv(X' * X)
TrZSZ = 0
for i in 1:p
    TrZSZ += Z[:, i]' * (Z[:, i] - X * InvXpX * X' * Z[:, i])
end

ZpZ = zeros(p)
for i in 1:p
    ZpZ[i] = Z[:, i]' * Z[:, i]
end

XpX = zeros(rX)
for i in 1:rX
    XpX[i] = X[:, i]' * X[:, i]
end

B = InvXpX * X' * Y
SY = Y - X * B
U = zeros(size(Z, 2), k)

Vy = diag((SY' * SY) / (n - rX))
Ve = copy(Vy) * 0.5
Vg = diagm(Ve ./ TrZSZ)
tilde = Z' * SY

convergence = 1.0
E = copy(SY)
iter = 0
maxit = 100

while convergence > 1e-8 && iter < maxit
    # Store current Beta
    U_base = copy(U)
    iter += 1

    iVe = 1.0 ./ Ve
    iVg = inv(Vg)

    # Gauss-Seidel for random effects
    order = randperm(p)
    for i in order
        U0 = copy(U[i, :])
        RHS = iVe .* ((Z[:, i]' * E)' + (ZpZ[i] * U0))
        LHS = diagm(iVe .* ZpZ[i]) + iVg
        U[i, :] = (LHS \ RHS)'
        E -= Z[:, i] * (U[i, :] - U0)'
    end

    # Update fixed effects
    B_tmp = InvXpX * X' * E
    B += B_tmp
    E -= X * B_tmp

    # Update VC
    Ve = diag((SY' * E) / (n - rX))
    Vg = tilde' * U / TrZSZ
    Vg = 0.5 * (Vg + Vg')

    # Update convergence
    convergence = maximum(sum((U - U_base).^2, dims=1))

    # Convergence
    println("iter $iter | conv $(log10(convergence))")
end

GEBV = Z * U

Accuracy = diag(cor(GEBV, TBV))

println(mean(Accuracy))