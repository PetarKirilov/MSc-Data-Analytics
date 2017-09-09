#ex1
  #ex1 from problem sheet in class
#a)
dpois(2,0.5*8) 
#b)
ppois(2,5*0.5)
  #ex3
1-ppois(16,12)
  #ex4
#a)
dexp(1,10/2)
#b)
dexp(2,10/2)
#c)
dpois(6,10/2)
#ex2
lambda <- 10
tmax <- 1
n <- 100
dt <- T/n
t <- (1:n)*dt
p <- lambda*dt
q <- 1-p
x <- sample(c(0,1),n,prob=c(q,p),replace=TRUE)
sum(x)
t[x==1]
event.t <- t[x==1]
plot(event.t, 0*event.t)
#ex3
plusone <- function(x) {
  y <- x+1
  return(y)
}
#ex4
