negloglike_t <- function(nu, R, u) {
  t_vals <- qt(u, nu)
  n <- nrow(t_vals)
  d <- ncol(t_vals)
  
  # Standardize: Z = T %*% solve(R)
  z <- t_vals %*% solve(R)
  
  nll <- -n * lgamma((nu + d) / 2) + n * d * lgamma((nu + 1) / 2) - n * (d - 1) * lgamma(nu / 2) +
    n * sum(log(diag(R))) +
    ((nu + d) / 2) * sum(log(1 + rowSums(z^2) / nu)) -
    ((nu + 1) / 2) * sum(rowSums(log(1 + t_vals^2 / nu)))
  
  return(nll)
}
