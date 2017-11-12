#ex1

#ex3
## Simulate the Galton -Walson process with offspring variable X ~ Poisson(mu)
gwn <- function(nGen ,mu=1,x0=1){
  x <- x0 # set the initial population size
  for( k in 1:nGen ){ # loop over nGen generations
    x <- rpois(1,lambda=x*mu) # sample the number of offsprings
    }
  return(x) # return the final population size
}

gwn(0, mu = 1.5)
dexp(x = gwn(0, mu = 1.5), )
