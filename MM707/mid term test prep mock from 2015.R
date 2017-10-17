#Q5 gamblers 2015
library(expm)
Q5 <- matrix(0, nrow = 8, ncol = 8)
Q5[1,1] <- 1; Q5[8,8] <- 1; Q5[2,c(1,3)] <- 0.5; Q5[3, c(2,5)] <- 0.5; Q5[4,c(3,7)] <- 0.5; Q5[5,c(4,8)] <- 0.5; Q5[6,c(5,8)] <- 0.5; Q5[7,c(6,8)] <- 0.5
eigQ5 <- eigen(Q5)$vectors
eigQ5ruin <- eigQ5[,1]
as.double(eigQ5ruin/(eigQ5ruin[1]))

#Q4 white/black balls in urn 2015
ball <- matrix(0, nrow = 4, ncol = 4)
ball[1,2] <- 1; ball[2,1] <- 1/9; ball[2,c(2,3)] <- 4/9; ball[3,c(2,3)] <- 4/9; ball[3,4] <- 1/9; ball[4,3] <- 1
ball3 <- ball %^% 3
ball3[1,4]
eigball <- eigen(t(ball))$vectors[,1]
eigball/sum(eigball)

#Q2

