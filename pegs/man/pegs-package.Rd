\name{pegs-package}
\alias{pegs-package}
\alias{Y}
\alias{X}
\alias{simu}
\docType{package}
\title{\packageTitle{pegs}}
\description{\packageDescription{pegs}}
\author{Alencar Xavier}
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