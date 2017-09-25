set.seed(16032017)
ss <- 300
norm.1d <- rnorm(ss)
hist(norm.1d)
norm.graphs <- function(x){hist(x,main=paste("Histogram of",deparse(substitute(x),500),"with
Sturge breaks"));hist(x,breaks="Scott",main=paste("Histogram
                                                  of",deparse(substitute(x),500),"with Scott breaks"));plot(density(x,bw="SJ"));boxplot(x);qqno
  rm(x);qqline(c(x),col="red");}
ss3 <- floor(ss/3)
mix3.1d <- c(rnorm(ss3,mean=-4),rnorm(ss3,mean=0),rnorm(ss-2*ss3,mean=4))

as.data.frame(mix3.1d)
dset1.num <- c(dset1$V1, dset1$V2, dset1$V3)
norm.graphs(dset1.num)
library(rgl)
plot3d(dset1,asp=1)
plot3d(dset2,asp=1)

