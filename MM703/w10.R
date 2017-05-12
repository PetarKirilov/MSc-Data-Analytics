#Week 10
z.conf.int<-function(mu,sigma,n,level=0.95){ margin<-sigma*qnorm(1-(1-level)/2)/sqrt(n);
  c(mu-margin,mu+margin)}
t.conf.int<-function(mu,sigma,n,level=0.95){ margin<-sigma*qt(1-(1-level)/2,n-1)/sqrt(n); 
  c(mu-margin,mu+margin)}
#Q5
z.conf.int(72,12,100)
#Q6
t.conf.int(25467,4980,100)
3672*t.conf.int(25467,4980,100, level = 0.99)
#Q7
z.conf.int(1,0.03,11)
mean(1.02,1.05,1.08,1.03,1.00,1.06,1.08,1.01,1.04,1.07,1)
z.conf.int(1.02,0.03,11)
#not fininshed
