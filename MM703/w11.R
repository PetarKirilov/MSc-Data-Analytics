#Lecture11
z.conf.int<-function(mu,sigma,n,level=0.95){ margin<-sigma*qnorm(1-(1-level)/2)/sqrt(n);
  c(mu-margin,mu+margin)}
t.conf.int<-function(mu,sigma,n,level=0.95){ margin<-sigma*qt(1-(1-level)/2,n-1)/sqrt(n); 
  c(mu-margin,mu+margin)}
z.conf.int(322.4,5,30)
phyper(1,6,6,6)
?pt
