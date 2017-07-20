#exam prep
#Q1
colony <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Revision for test/colony2.csv")
cor(colony)
#after checking the correlation, 0.9 correlation is quite strong, thus a regression is appropriate
plot(colony)
#relationship is not linear, thus we need to linearize it by doing some transformations to the data 
plot(colony$Minutes, log(colony$Size))
#if we take the log of the size, the relationship becomes more linear
abline(log(colony$Size)~colony$Minutes)
summary(colony.lm <- lm(log(Size)~Minutes, data = colony))
#R^2 is 98.92% which concludes that ~99% of the variability in the outcome is explained by the predictor. THe intercept is 3.511 and the beta is 0.0145.
#The model is significant, the p-value is way below the treshold of 5%.
plot(colony.lm)
#residuals look ok. It looks like there is some pattern but due to file size
shapiro.test(colony.lm$residuals)
#residuals are normally distributed
exp(predict(colony.lm, data.frame(Minutes=2.5*60), interval = "conf"))
