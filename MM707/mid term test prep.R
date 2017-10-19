#Q5 test 2015-16
#c
library(expm)
AL <- matrix(nrow = 8,ncol = 8,0)
AL[1,1] <- 1; AL[8,8]<-1;AL[2,c(1,3)] <- 0.5; AL[3,c(2,5)]<-0.5; AL[4,c(3,7)] <- 0.5; AL[5,c(4,8)] <- 0.5; AL[6,c(5,8)] <- 0.5; AL[7,c(6,8)] <- 0.5
eigAL <- eigen((AL))
lamAL <- eigAL$values
vecAL <- eigAL$vectors
vAL <- vecAL[,1]
vAL <- as.double(vAL)
vAL <- vAL/(vAL[1])
vAL[2]
#this is the prob that she is ruined given that she starts with 1 coin

#Q5 2016-17
AL16 <- matrix(nrow = 6, ncol = 6, 0)
AL16[1,1] <- 1; AL16[6,6] <- 1; AL16[2, c(1,3)] <- 1/2; AL16[3,2] <- 3/4; AL16[3,4] <- 1/4; AL16[4,3] <- 7/8; AL16[4,5] <- 1/8; AL16[5,4] <- 15/16; AL16[5,6] <- 1/16
eigAL16 <- eigen((AL16))
vAL16 <- eigAL16$vectors[,2]
vAL16 <- as.double(vAL16)
vAL16 <- vAL16/vAL16[6]
#we select the second vector of eigen values as it has the probability to win, whereas the first one has the probability to lose
