#W06
#q1

RuinChance <- function(k, a, p){
  AM <- matrix(0, nrow = a, ncol = a)
  AM[row(AM)-1 == col(AM)] <- (1-p); AM[row(AM)+1==col(AM)] <- p; AM[1,1] <- 1; AM[a,a] <- 1; AM[1,2] <- 0; AM[2, 1] <- 0
  eigAM <- eigen(t(AM))
  lamAM <- eigAM$values
  vecAM <- eigAM$vectors
  vAM <- vecAM[,1]
  vAM <- vAM/sum(vAM)
  
}


RuinChance <- function(k, a, p){
  if(p != 0.5){
    uk <- (((1-p)/p)^k-((1-p)/p)^a)/(1-((1-p)/p)^a)
    dk <- (1/(1-2*p))*((k-(a*(1-((1-p)/p)^k)))/(1-((1-p)/p)^a))
    return(paste(c("The probability of ruin is"),round(uk, digits = 4), c("and the duration of the game is"), round(dk)))
    } else {
    uk <- (a-k)/a
    dk <- k*(a-k)
    return(paste(c("The probability of ruin is"), round(uk, digits = 4), c("and the duration of the game is"), round(dk)))
    }
}
RuinChance(5, 100, 0.6)
#for the below, if the duration of the game looks wrong, calculate it for the perspective of the dealer.
RuinChance(10, 80, 0.55)
RuinChance(40, 50, 0.5)

#Q2




AM <- matrix(0, nrow = 10, ncol = 10)
AM[1,1] <- 1; AM[10, 10] <- 1
AM[row(AM)+1 == col(AM)] <- 4
AM[row(AM)-1 == col(AM)] <- 5
AM[2,1] <- 0; AM[1,2] <- 0