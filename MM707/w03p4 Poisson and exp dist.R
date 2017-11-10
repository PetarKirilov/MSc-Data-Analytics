#
# simulates the Poisson process with the given rate
# returns the time of the first event
#
first.time <- function(rate){
#
# set up problem parameters
#
lambda <- rate                # event rate
T <- 20/rate                  # total simulation time (to fit approx 20 events)
n <- 1e4                      # number of points in time mesh
dt <- T/n                     # step size of the time mesh
t <- (1:n)*dt                 # the time mesh
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
