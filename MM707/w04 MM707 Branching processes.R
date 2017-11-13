#ex2
n <- rpois(n = 1, lambda = 300)
y1 <- sample(x=1:5, 1, prob = c(0.6, 0.3, 0.05, 0.04, 0.01))
y <- sample(x=1:5, n, prob = c(0.6, 0.3, 0.05, 0.04, 0.01), replace = TRUE)
sum(y)
#simulate the composite Poisson process
rcpois <- function(x, prob=1, lambda=1){
  n <- rpois(1, lambda)
  events <- sample(x,n,prob=prob, replace = TRUE)
  total <- sum(events)
  return(total)
}
rcpois(1:5, prob = c(0.6,0.3,0.05,0.04,0.01), lambda = 300)
n <- 1e6
z <- replicate(n, expr = rcpois(1:5, prob = c(0.6,0.3,0.05,0.04,0.01), lambda = 300))
mean(z)
#theoretical mean is 468 and result with the current random variables is 469.53, so it is pretty close
var(z)
#theoretical var is 942 and the result is 930.77, which is not so close

#ex3
#simulate the Galton-Walson process with offspring variables X ~ Poisson(mu)
gwn <- function(nGen, mu = 1, x0 = 1){
  x <- x0 #set the initial population size
  for(k in 1:nGen){ #loop over nGen generations
    x <- #sample the true number of offsprings
  }
  return(x) #return the final population size
}