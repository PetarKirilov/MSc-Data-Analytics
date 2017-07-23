library(MASS)
set.seed([redacted])
student.id <- c("24","33","88","78","75","62","47","56","53","57","01","81","52","73")
n <- 1000
[redacted]
sigma <- matrix([redacted],nrow=3)
for (id in student.id){
  mixture <- rbind([redacted])
  simple <- mvrnorm(n,mu=colMeans(mixture),Sigma=var(mixture),empirical=TRUE)
  mixt <- sample.int(2,1)
  write.csv(mixture,file=paste("sample-",id,"-",mixt,".csv",sep=""))
  write.csv(simple,file=paste("sample-",id,"-",3-mixt,".csv",sep=""))
  write(paste("Mixture is file",id,mixt),file=paste("answer",id,".txt",sep=""))
}