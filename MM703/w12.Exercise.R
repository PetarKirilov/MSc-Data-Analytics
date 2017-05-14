set.seed(20170112)
n <- 100
s.mean <- 4
s.std <- sqrt(s.mean)
s.pois <- jitter(rpois(n,s.mean))
s.norm <- rnorm(n,s.mean,s.std)
s.binom <- jitter(rbinom(n,s.mean^2/(s.mean-s.std),1-s.std/s.mean))
s.mix <- sample(c(rnorm(n/2,s.mean,sqrt(2*s.std^2-1)),rnorm(n-n/2,s.mean,1)))
write.table(data.frame(s.norm,s.mix,s.pois,s.binom),file="Datasets.csv",row.names=FALSE,col.names=c("First Sample","Second Sample","Third Sample","Fourth Sample"),sep=",")

