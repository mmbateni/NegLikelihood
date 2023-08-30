negloglike_t <- function(nu, R, u) {
  # Compute negative log-likelihood for a t copula at nu and R = chol(Rho)
  t <- qt(u, nu)
  n <- nrow(t)
  d <- ncol(t)
  R <- R / sqrt(colSums(R^2))
  
  tRinv <- t / R
  nll <- - n * lgamma((nu + d) / 2) + n * d * lgamma((nu + 1) / 2) - n * (d - 1) * lgamma(nu / 2) +
    n * sum(log(abs(diag(R)))) +
    ((nu + d) / 2) * sum(log(1 + rowSums(tRinv^2) / nu)) -
    ((nu + 1) / 2) * sum(rowSums(log(1 + t^2 / nu)))
  
  return(nll)
}
