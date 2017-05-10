#Lecture 9 Continuous dist
#Lec example
1-pnorm(12, 12, 3)
sum(dnorm(9:14, 12, 3))
pnorm(15,12,3)-pnorm(9, 12,3)
pnorm(15,12,3)
qnorm(0.75,12,3)
#Q1
dpois(0,1)
dpois(1,1)
ppois(1,1)
ppois(2,1)
1-ppois(1,1)
dpois(0,2)
dpois(1,2)
ppois(1,2)
ppois(2,2)
1-ppois(1,2)
#Q2
dpois(3,1)
ppois(2,1)
1-ppois(4,1)
ppois(3,1)-ppois(0,1)
#Q3
5/20 #gives 0.25 psm
5/20*16 #same as variance due to poisson properties
dpois(0,4)
dpois(6,4)
ppois(6,4)
1-sum(dpois(0:5,4)) #or 1-ppois(5,4)
#Q4
dpois(0,15)
#Q5
1-pexp(70,0.2)
#Q6
pexp(1.5,0.4)-pexp(1,0.4)
dexp(1,0.4)
dpois(0,0.4) #or 1-pexp(1,0.4)
1-ppois(1,0.4)
#Q7
dpois(3,3.5)
1-ppois(2,3.5)
dpois(9,3.5*3)
dpois(3,3.5)^3 #as the breakdown happens independently
pexp(1,3.5)
pexp(96/168,3.5)-pexp(12/168,3.5)
pexp(4/7,3.5)-pexp(3/7,3.5)
#Q8
pexp(100,1/1000, FALSE)*pexp(100,1/300,FALSE)*pexp(100,1/150,FALSE)
#Q9
1-pexp(40,1/80, FALSE)^5
#Q10
punif(1,0,2)
1-punif(1.5,0,2)
punif(1.5,0,2)-punif(0.5,0,2)
#Q11
1/dunif(230,227.5,232.5)
punif(231.6,227.5,232.5)-punif(229.3,227.5,232.5)
#for a continuous dist the prob to take ANY fixed value is 0
1-punif(130,227.5,232.5)
#Q12
1-pnorm(8,6,2)
pnorm(8,6,2)
pnorm(8,6,2)-pnorm(6,6,2)
pnorm(0,6,2)
1-qnorm(0.25,6,2)
1-qnorm(0.75,6,2)
#Q13
pnorm(500,450,100)-(pnorm(400,450,100))
1-pnorm(480,450,100)
qnorm(0.1,450,100,FALSE)
#Q14
18-0.6*qnorm(0.2)
