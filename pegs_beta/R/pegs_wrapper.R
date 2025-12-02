pegs <- function(Y, X, maxit = 100, logtol = -4.0, NonNegativeCorr = FALSE) {
  # Type check
  if (!is.matrix(Y)) Y <- as.matrix(Y)
  if (typeof(Y) == "integer") storage.mode(Y) <- "numeric"
  if (typeof(X) == "integer") storage.mode(X) <- "numeric"
  # Call the C++ function
  res <- .Call("PEGS_lapack",
               as.matrix(Y),
               as.matrix(X),
               as.integer(maxit),
               as.double(logtol),
               as.logical(NonNegativeCorr))
  # Add dimension names to the output for clarity
  dimnames(res$hat) <- dimnames(Y)
  dimnames(res$b) <- list(colnames(X), colnames(Y))
  dimnames(res$GC) <- list(colnames(Y), colnames(Y))
  names(res$h2) <- colnames(Y)
  names(res$mu) <- colnames(Y)
  return(res)
}
