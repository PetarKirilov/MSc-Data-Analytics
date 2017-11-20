library(expm)
P <- matrix(0, nrow = 3, ncol = 3)
P[2,3] <- 8
Pt <- t(P)

#ex1
G <- matrix(0, nrow = 2, ncol = 2)
G[1,1] <- 0.4; G[1,2] <- 0.6; G[2,1] <- 0.6; G[2,2] <- 0.4
#fill in the matrix
Gsq <- G %^% 2
#this is the square probability matrix, showing the 2-step transition probability matrix
#c
#as there are three steps, the probability matrix needs to be raised to the third power
Gcub <- G %^% 3
#the above gives us the probabilies after three steps, i.e. the probabilities for Thursday
#thus the probability to start with a black tie and end up with a black tie is contained in the 1st element [1,1]
Gcub[1,1]

#ex2
#A
#the probability matrix is:
R <- matrix(0, nrow = 2, ncol = 2)
R[1,1] <- 0.8; R[1,2] <- 0.2; R[2,1] <- 0.4; R[2,2] <- 0.6
#B
#If we assume today is Thursday
Rcub <- R %^% 3
Rcub[1,2]
#the chance is 0.312
#C
R4 <- R %^% 4
R4[1,2]
#the chance is 0.3248
#D
R14 <- R %^% 14
R14[1,2]
#the chance is 0.33
#E
fut <- as.Date("2044/07/01")
steps <- fut - Sys.Date() 
Rfut <- R %^% as.numeric(steps)
#the chance of rain on that date is:
Rfut[1,2]

#ex2
#a
H <- matrix(0, ncol = 3, nrow = 3)
H[1,2] <- 0.5; H[1,3] <- 0.5; H[2,1] <- 0.5; H[2,3] <- 0.5; H[3,1] <- 0.5; H[3,2] <- 0.5
#b
#the three step matrix is below
H3 <- H %^% 3
#c
#in order to find the steady state, we need to find the eigen values:
eig <- eigen(t(H))
#eigen value of [0,1 is the stationary state]
lam <- eig$values
vec <- eig$vectors
v <- vec[,1]
v <- v/sum(v)
#we had to scale it as the sum needs to add up to 1
v
#the above is the steady state

#ex4
#a
#need to create the matrix of the transition matrix
L <- matrix(0, ncol = 5, nrow = 5)
L[1,1:2] <- 0.5; L[1:5,1] <- 0.5; L[2,3] <- 0.5; L[3,4] <- 0.5; L[4,5] <- 0.5; L[5,5] <- 0.5
#b
Lsq <- L %^% 2
#above is the 2-step transition probability matrix
#c
eig1 <- eigen(t(L))
lam1 <- eig1$values
vec1 <- eig1$vectors
v1 <- vec1[,1]
v1 <- v1/sum(v1)
as.double(v1)
#the above v1 vector is the steady state of the Markov Chain

#ex5
#a and b
S <- matrix(0, ncol = 5, nrow = 5)
S[1,5] <- 1; S[2,1] <- 1; S[3,1:2] <- 0.5; S[4,1:3] <- 1/3; S[5,1:4] <- 0.25
#the above is the transition matrix for the particular Markov chain
#c
eig2 <- eigen(t(S))
lam2 <- eig2$values
vec2 <- eig2$vectors
v2 <- vec2[,1]
v2 <- v2/sum(v2)
as.double(v2)
#the above v2 is the steady state of the Markov chain