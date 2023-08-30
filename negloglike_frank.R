negloglike_frank <- function(alpha, u) {
  # C(u1,u2) = -(1/alpha)*log(1 + (exp(-alpha*u1)-1)*(exp(-alpha*u1)-1)/(exp(-alpha)-1))
  expau <- exp(alpha * u)
  sumu <- rowSums(u)
  if (abs(alpha) < 1e-5) {
    logy <- 2 * alpha * prod(u - 0.5, 2) # -> zero as alpha -> 0
  } else {
    logy <- log(-alpha * expm1(-alpha)) + alpha * sumu -
      2 * log(abs(1 + exp(alpha * (sumu - 1)) - sum(expau, 2)))
  }
  nll <- -sum(logy)
    # Return approximate 2nd derivative of the neg loglikelihood,
    # using -E[score^2] = E[hessian]
    if (abs(alpha) < 1e-5) {
      dlogy <- 2 * prod(u - 0.5, 2)
    } else {
      dlogy <- 1 / alpha + 1 / expm1(alpha) + sumu -
        2 * ((sumu - 1) * exp(alpha * (sumu - 1)) - rowSums(u * expau)) /
        (1 + exp(alpha * (sumu - 1)) - sum(expau, 2))
    }
    d2 <- sum(dlogy^2)
    return(list(nll = nll, d2 = d2, dlogy = dlogy))
}
