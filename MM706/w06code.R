sum(sample(0:1, 2)) #because we do not have replace = TRUE, it does not assume that you have equal probabilities
replicate(10, sum(sample(0:1,2, replace = TRUE)))
#c
n <- 10^6
test1 <- sample(0:1,n, replace = TRUE)
test2 <- sample(0:1,n, replace = TRUE)
est <- table(test1 + test2)/n
#the estimates are: for 0 is 0.25; for 1 is 0.499; for 2 is 0.25
ptrue <- c(1, 2, 1)/4
max(abs(est-ptrue))

#d
test <- replicate(n, sum(sample(0:1,1024, replace = TRUE)))
est <- table(test)/n
hist(test)

#prob2
n <- 10^3
x <- runif(n)
y <- runif(n)
d <- sqrt((x-0.5)^2+(y-0.5)^2)
table(d >= 0.5)/n
pi/4

