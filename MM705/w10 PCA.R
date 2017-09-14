#ex1
#gen trivariate dset from normal dist
library(MASS)
ss <- 1000
z.3d <- c(0,0,0)
d3.dset <- matrix(c(1,0.8,0.8,0.8,1,0.8,0.8,0.8,1),nrow=3)
norm.3d <- mvrnorm(ss,mu=z.3d,Sigma=d3.dset)
pairs(norm.3d, asp = 1)
PCA.norm.3d <- princomp(norm.3d)
PCA.norm.3d
pairs(PCA.norm.3d$scores, asp = 1)

d3.dset <- matrix(c(1,0.9,0.1,0.3,1,0.5,0.3,0.1,1),nrow=3)
normdset <- mvrnorm(ss, mu = z.3d, Sigma = d3.dset)
pairs(normdset, asp = 1)
PCA.norm.3d1 <- princomp(normdset)
pairs(PCA.norm.3d1$scores, asp = 1)

#ex2
air <- Airquality
head(air)
pairs(air)
air.PCA <- princomp(~Ozone+Solar.R+Wind+Temp, data=air)
summary(air.PCA, loadings = TRUE)
pairs(air.PCA$scores, asp = 1)
plot(air.PCA$sdev^2,type="b",main="Scree plot",ylab="Eigenvalue")
