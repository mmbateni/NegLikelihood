negloglike_clayton <- function(alpha, u)
  {
  # C(u1,u2) = (u1^(-alpha) + u2^(-alpha) - 1)^(-1/alpha)
  powu <- u^(-alpha)
  lnu <- log(u)
  logC <- (-1/alpha) * log(rowSums(powu) - 1)
  logy <- log(alpha + 1) + (2 * alpha + 1) * logC - (alpha + 1) * rowSums(lnu)
  nll <- -sum(logy)
  # Return approximate 2nd derivative of the neg loglikelihood,
  # using -E[score^2] = E[hessian]
  dlogy <- 1 / (1 + alpha) - logC / alpha +
  (2 + 1 / alpha) * rowSums(powu * lnu) / (rowSums(powu) - 1) - rowSums(lnu)
  d2 <- sum(dlogy^2)
  return(list(nll = nll, d2 = d2))
}
