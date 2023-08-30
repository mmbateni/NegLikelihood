negloglike_gumbel <- function(alpha, u) {
  # C(u1,u2) = exp(-((-log(u1))^alpha + (-log(u2))^alpha)^(1/alpha))
  v <- -log(u) # u is strictly in (0,1) => v strictly in (0,Inf)
  v <- t(apply(v, 1, sort)) # Sort rows
  vmin <- v[, 1]
  vmax <- v[, 2]
  logv <- log(v)
  nlogC <- vmax * (1 + (vmin / vmax)^alpha)^(1 / alpha)
  lognlogC <- log(nlogC)
  logy <- log(alpha - 1 + nlogC) - nlogC +
    rowSums((alpha - 1) * logv + v) + (1 - 2 * alpha) * lognlogC
  nll <- -sum(logy)
    # Return approximate 2nd derivative of the neg loglikelihood,
    # using -E[score^2] = E[hessian]
    dnlogC <- nlogC * (-lognlogC + rowSums(logv * v^alpha) / rowSums(v^alpha)) / alpha
    dlogy <- (1 + dnlogC) / (alpha - 1 + nlogC) - dnlogC +
      rowSums(logv) + (1 - 2 * alpha) * dnlogC / nlogC - 2 * lognlogC
    d2 <- sum(dlogy^2)
    return(list(nll = nll, d2 = d2))
}
