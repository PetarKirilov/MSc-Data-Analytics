first.time <- function(rate){
#
# set up problem parameters
#
lambda <- rate                # event rate
T <- 1                        # total simulation time
n <- 10000                    # number of points in the mesh
dt <- T/n                     # time step
t <- (1:n)*dt                 # mesh on time, the same as t <- seq(dt,T,by=dt)
#
# simulate elementary events
#
p1 <- lambda*dt               # probability of an event
p0 <- 1-p1                    # probability of no event
x <- sample(c(0,1),           # vector of outcomes (0=no event, 1=event)
            n,                # number of elements to generate
            prob=c(p0,p1),    # vector of probabilities of outcomes
            replace=TRUE)     # to reuse elements from the bin
#
# find event times
#
event.t <- t[x==1]            # times when an event is recorded
t1 <- min(event.t)            # time of the first event
return(t1)
}
replicate(1e4, first.time(10))
plot(density(replicate(1e4, first.time(10))))
te <- seq(-0.3,1.3,length=100)
plot(density(dexp(te, 10)))
