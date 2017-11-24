#exponential smoothing
library(forecast)
d <- c(3,4,41,10,9,86,56,20,18,36,24,59,82,51,31,29,13,7,26,19,20,103,141,145,24,99,40,51,72,58,94,78,11,15,17,53,44,34,12,15,32,14,15,26,75,110,56,43,19,17,33,26,40,42,18,24,69,18,18,25,86,106,104,35,43,12,4,20,16,8)
mod1 <- HoltWinters(d[1:40], alpha=0.1, beta=FALSE, gamma=FALSE)
plot(predict(mod1, n.ahead=30))
ses(d[1:40], h=30, alpha=0.1, initial="simple")

#ex1
library(forecast)
prac1 <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM707 - Stochastic Methods and Forecasting OPTIONAL SEM 2 20CR/w07 Exponential Smoothing/practical1.csv")
prac1.predict <- ses(prac1, h = 5, alpha = 0.1, initial = "simple")
plot(prac1.predict, plot.conf = FALSE)
