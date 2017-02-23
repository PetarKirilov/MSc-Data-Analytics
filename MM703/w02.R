dbinom(4, 10, 0.3)
#lec2
#ex1
#i
dbinom(2, 10, 0.1)
#ii
sum(dbinom(0:2, 10, 0.1))
pbinom(2, 10, 0.1)
#iii
sum(dbinom(3:10, 10, 0.1))
1-pbinom(2, 10, 0.1)
#iiii
sum(dbinom(2:4, 10, 0.1))
pbinom(4, 10, 0.1) - pbinom(1,10,0.1)
#ex2
#c
dbinom(0,2,0.03)
dbinom(1,2,0.03)
1+dbinom(2,2,0.03)
#ex3
#b
#i
dbinom(3, 5, 0.24)
#ii
sum(dbinom(2:4, 5, 0.24))
#iii
pbinom(2, 5, 0.24)
#ex4

#i
sum(c(0,1,2,3)*c(0.125,0.375,0.375,0.125))
#ii
sum(c(20,25,30,35)*c(0.2,0.15,0.25,0.4))
#iii
sum(c(0,1,2,3,4,5)*c(0.18,0.39,0.24,0.14,0.04,0.01))
#iiii
sum(c(-100,0,50,100,150,200)*c(0.1,0.2,0.3,0.25,0.1,0.05))
#ex11
#a
mean(rbinom(100,3,0.5))
mean(rbinom(1000,3,0.5))
#ex12
sum(sample(0:1, 20, TRUE, c(18/38, (1-18/38))))

