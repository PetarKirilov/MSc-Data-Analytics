x <- runif(10^5, 0, 1)
y <- runif(10^5, 0, 1)
z <- runif(10^5, 0, 1)
mean(x*y^2*z^3)
#o.o415
mean(x+y<z)
#0.168
mean(pmin(x,y,z))
#0.249
#####
#q2
ppois(2, 30/6)
#o.12
1-pexp(3,30/60)


func <- function(rate,tmax,prob){
  t <- 0
  z <- 0
  n <- length(prob)
  while(t<tmax){
    dt <- rexp(1,rate)
    y <- sample(1:n,1, prob=prob)
    t <- t+dt
    if(t<tmax) z<-z+y
  }
  return(z)
}
lin <- replicate(10^4, func(10,0.5,c(0.5,0.3,0.15,0.05)))
mean(lin>10)
