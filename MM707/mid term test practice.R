nc <- function(rate, tmax, prob) {
  t <- 0; z <- 0
  n <- length(prob)
  while(t<tmax){
    dt <- rexp(1, rate)
    y<- sample(1:n,1,prob=prob)
    t <- t+dt
    if(t<tmax) z<- z+y
  }
  return(z)
}