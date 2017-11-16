# Simulate the composite Poisson process

rcpois = function(x,prob=1,lambda=1){
 n=rpois(1,lambda)
 events=sample(x,n,prob=prob,replace=TRUE)
 total=sum(events)
 return(total)
}
