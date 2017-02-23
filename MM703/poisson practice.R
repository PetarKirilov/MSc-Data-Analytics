#Poisson practice
#if the mean is 10 and we are interested in the probability of 5 cars to arrive in 15 mins (the mean is 10 cars for 15 minutes):
dpois(5,10)
#the prob of one arrival in the next 3 minutes (the mean is 2 cars per three minutes)
dpois(1,2)
#the probability of finding no defects in a 3 km highway (the mean is 2 defects per km)
dpois(0,6)
#ex28,p177
dpois(2,3)
dpois(1,3)
1-sum(dpois(0:1,3))

#Hypergeometric distribution
#12 fuses in a box - a person draws 3 of the box; if the box has exactly 5 defect fuses, what is the prob that 1 defective is drawn:
dhyper(1,5,7,3)
#you need to subtract the drawn fuses from the total amount to be correct calculation

#same example as above but this time, what is the probability that AT LEAST 1 defective fuse is found?
#to calculate, first calculate the prob of finding 0 defective and subtract it from 1
1-dhyper(0,5,7,3) 
#ex33,p180
dhyper(1,3,7,4)
#ex34
dhyper(3,4,11,10)

#Uniform dist practice
#random var x is uniformly distributed between 10 ans 20; find:
#a P[X<15]
sum(dunif(x = 10:14,min = 10, max = 20))
#b[12<=X<=18]
punif(18, 10, 20)-punif(12,10,20)

#ex6
punif(12.05, 11.975, 12.1)-punif(12, 11.975, 12.1)


#Normal dist
#prob of Z=1 where mean=0 sd=1
pnorm(1,0,1)
#P[-0.5<=Z<=1.25]
pnorm(1.25,0,1)-pnorm(-0.5,0,1)
#P[-1<=Z<=1]
pnorm(1,0,1)-pnorm(-1,0,1)
#P[Z=>1.58]
1-pnorm(1.58,0,1)
#P[10<=Z<=14] where mean=10, sd=2
pnorm(14,10,2)-pnorm(10,10,2)
#[P=>40000] where mean = 36500, sd = 5000
1-pnorm(40000, 36500, 5000)
#[P0<=Z<=1] where mean=0, sd=1
pnorm(1,0,1) - pnorm(0,0,1) 
#P[0<=Z<=1.5]
pnorm(1.5,0,1)-pnorm(0,0,1)
#P[0<Z<2]
pnorm(2,0,1) - pnorm(0,0,1) 
#P[0<Z<2.5]
pnorm(2.5,0,1) - pnorm(0,0,1) 
#P[0<=Z<=0.83]
pnorm(0.83,0,1) - pnorm(0,0,1) 
#P[-1.57<=Z<=0]
pnorm(0,0,1) - pnorm(-1.57,0,1) 
#P[Z>0.44]
1-pnorm(0.44,0,1) 
#P[Z=>-0.23]
1-pnorm(-0.23,0,1) 
#P[Z<1.2]
pnorm(1.2,0,1) 
#P[Z<=-0.71]
pnorm(-0.71,0,1)
#12 successes in 100 trials, p=0.1
dbinom(12,100, 0.1)

?dexp
1-dexp(12, rate=12)
