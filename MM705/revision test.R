#Revision for test
#ex1
colony <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/colony.csv")
plot(colony)
lines(colony) #line connecting the instances
abline(lm(colony$Count~colony$Time), col="red") # regression line (y~x) 
lines(lowess(colony$Time,colony$Count), col="blue") # lowess line (x,y)
summary(colony.model <- aov(Count~Time, data = colony)) 
summary(lm(Count~Time, data = colony))
