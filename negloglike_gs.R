negloglike_gs <- function(rhohat, u) {
  x <- qnorm(u[, 1])
  y <- qnorm(u[, 2])
  
  F <- -sum(log((1 - rhohat^2)^(-0.5) * exp((x^2 + y^2) / 2 + (2 * rhohat * x * y - x^2 - y^2) / (2 * (1 - rhohat^2)))))
  
  return(F)
}
