\name{pegs}
\alias{pegs}
\title{Multivariate Ridge Regression}
\usage{
pegs(Y, X, maxit = 100, logtol = -4, NonNegativeCorr = TRUE)
}
\arguments{
\item{Y}{A numeric matrix (n x k) of phenotypes, where n is the number of
individuals and k is the number of traits. Missing values should be NA.}

\item{X}{A numeric matrix (n x p) of genotypes/markers, where p is the
number of markers.}

\item{maxit}{An integer specifying the maximum number of iterations.}

\item{logtol}{A numeric value for the convergence tolerance on a log10 scale.
The algorithm stops when the log10 of the sum of squared changes in
coefficients is less than this value.}

\item{NonNegativeCorr}{A logical value. If TRUE, the off-diagonal elements
of the genetic variance-covariance matrix are constrained to be non-negative.}
}
\value{
A list containing the following components:
\describe{
  \item{mu}{Vector of intercepts for each trait.}
  \item{b}{Matrix of marker effects (p x k).}
  \item{hat}{Matrix of predicted values (n x k).}
  \item{h2}{Vector of heritability estimates for each trait.}
  \item{GC}{The genetic correlation matrix (k x k).}
  \item{bend}{The inflation factor added to the diagonal of the genetic
    variance matrix to ensure it is positive definite.}
  \item{numit}{The number of iterations performed.}
  \item{cnv}{The final convergence value.}
}
}
\description{
This function implements the Pseudo-Expectation Gauss-Seidel algorithm, used in multi-trait genomic prediction models.
}
