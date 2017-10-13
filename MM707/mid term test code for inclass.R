#TEST 16-17
library(expm)
#Q1
x <- runif(10^7, 0, 1)
y <- runif(10^7, 0, 1)
z <- runif(10^7, 0, 1)
#a
mean(x*y^2*z^3)
#b
mean(x+y<z)
#c
mean(pmin(x,y,z))

#Q2
#a
ppois(2,5)
#b
1-pexp(3,0.5)

#Q4
#a
M <- matrix(0, ncol = 4, nrow = 4)
M[1,4] <- 1; M[2,3] <- 0.3; M[2,4] <- 0.7; M[3,2] <- 0.3; M[3,3] <- 0.7; M[4,1] <- 0.3; M[4,2] <- 0.7
#the above is the transition matrix
#b
Msq <- M %^% 4
Msq[3,1]
#c
eigM <- eigen(t(M))
lamM <- eigM$values
vecM <- eigM$vectors
vM <- vecM[,1]
vM <- vM/sum(vM)
#the steady state of the current Markov chain is above. Thus, the steady state when he gets wet is the first argument times the probability that it rains
vM[1]*0.7



#Q2
#lambda is 30 ph
y.prob <- c(0.5,0.3,0.15, 0.05)
1 - ppois(3, 30/1/6)




#TEST 2015-16
#Q1
#a
x <- runif(10^7, 0, 1)
y <- runif(10^7, 0, 2)
mean(x*y^2-x^2*y)
#b
mean(y^2 < x)
#c
z <- pmin(x,y)
mean(z)
#Q2
#a
1/(15*0.2)
#the average time is 20 minutes, as for the Poisson process of cars with one driver the lambda is 3, thus the average time is 1/3 which is 20 mins
#b

#TEST 14-15
#Q2
#a
dpois(3,5*2)
#b
dpois(0,1/6*5)
#c
1-ppois(1,4*2)
#d
dpois(5,4)*dpois(1,1)
#as the events are independent, we can multiply their probabilities to find the oveall probability

#Q4
#a
N <- matrix(0, ncol = 4, nrow = 4)
N[1,4] <- 1; N[2,3] <- 0.4; N[2,4] <- 0.6; N[3,2] <- 0.4; N[3,3] <- 0.6; N[4,1] <- 0.4; N[4,2] <- 0.6
#b
N10 <- N %^% 10
N10[3,1]
#c
eigN <- eigen(t(N))
lamN <- eigN$values
vecN <- eigN$vectors
vN <- vecN[,1]
vN <- vN/sum(vN)
#the above is the steady state of the Markov chain. The state that he gets wet is the first agrument times the probability that it will rain
vN[1]*0.6

#Test Speciment 2015-16

#Q2
#a
60/(15*0.2)




#Q4
#a
B <- matrix(0, nrow = 4, ncol = 4 )
B[1,2] <- 1; B[2,1] <- 1/9; B[2,2:3] <- 4/9; B[3,2:3] <- 4/9; B[3,4] <- 1/9; B[4,3] <- 1
#b
B3 <- B %^% 3
B3[1,4]
#this is the probability to have 3 black balls in the left urn
#c
eigB <- eigen(t(B))
lamB <- eigB$values
vecB <- eigB$vectors
vB <- vecB[,1]
vB <- vB/sum(vB)

#EXAM Specimen 2 2014
#Gabmblers ruin
#Q4
EX2 <- matrix(0, nrow = 8, ncol = 8)
EX2[1,1] <- 1; EX2[2,c(1,3)] <-0.5; EX2[3,c(1,5)] <- 1/4; EX2[3,3] <-2/4; EX2[4,4] <- 1; EX2[5,c(1,8)] <- 1/16; EX2[5,c(3,7)] <- 4/16; EX2[5,5] <- 6/16; EX2[6,6] <- 1; EX2[7,1] <- 1/64; EX2[7,3] <- 6/64; EX2[7,5] <- 15/64; EX2[7,7] <- 20/64; EX2[7,8] <- 22/64; EX2[8,8] <- 1
eigen(EX2)
#there are four eigen values, but two of them are for 3 and 5 that we defined as states that cannot be entered into. Thus, we only look at the first and the last eigen value which correspond to Alise's ruin or winning.
#the first eigen vector has 0 at the last state [8] which corresponds to her not winning at all, meaning that this eigenvector is for the gambler's ruin
#the other eigenvector corresponds to her winning as the initial state [1] is zero, meaning that she cannot win if she has zero coins initially
#we need the eigenvector showing the ruin, thus:
ruin <- eigen(EX2)$vectors[,1]
ruin <- ruin/ruin[1]
#from the rescaled eigenvector, we can deduce that the probability of Alice to be ruined if she started with one coin is:
ruin[2]
#d
ruin[2]*0+(1-ruin[2])*7 -1
#from the above equation, we can deduce that on average Alice loses 0.175 coins per game
