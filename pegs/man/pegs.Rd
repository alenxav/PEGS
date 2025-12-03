\name{pegs}
\alias{pegs}
\title{Pseudo-Expectation Gauss-Seidel}
\usage{
pegs(Y, X, maxit = 100, logtol = -4, NNC = FALSE)
}
\arguments{
\item{Y}{A numeric matrix (n x k) of dependent variables, where n is the number of
observations and k is the number of response variables. Missing values should be NA.}

\item{X}{A numeric matrix (n x p) of observations by parameters (i.e., markers), where p is the number of parameters.}

\item{maxit}{An integer specifying the maximum number of iterations.}

\item{logtol}{A numeric value for the convergence tolerance on a log10 scale.
The algorithm stops when the log10 of the sum of squared changes in
coefficients is less than this value.}

\item{NNC}{A logical value. If TRUE, imposes a constrain of non-negative correlations.}
}
\value{
A list containing the following components:
\describe{
  \item{mu}{Vector of intercepts for each response variable.}
  \item{b}{Matrix of regression coefficients (p x k).}
  \item{hat}{Matrix of predicted values (n x k).}
  \item{h2}{Vector of heritability estimates for each trait.}
  \item{GC}{The genetic correlation matrix (k x k).}
  \item{bend}{The inflation factor added to the diagonal of the genetic
    variance matrix to ensure it is positive definite.}
  \item{numit}{The number of iterations until convergence.}
  \item{cnv}{The final convergence value.}
}
}
\description{
This function implements the Pseudo-Expectation Gauss-Seidel algorithm, used in multi-trait genomic prediction models.
}
\examples{

# Load simulated data
data(test);

# Fit multivariate ridge regression
fit = pegs(Y,X);

# Heritability
print(fit$h2);

# Genetic correlations
print(fit$GC);

# Accuracy
print(mean(diag(cor(fit$hat,simu$tbv))));

}
