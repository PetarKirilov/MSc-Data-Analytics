library(car)
colony <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Revision for test/colony2.csv")
cor(colony)
#there is correlation between the variables, thus a regresson is applicable
plot(colony)
#the relationship is not linear, as it represents a smooth line, thus we need to make some transformations
plot(colony$Minutes, log(colony$Size))
#by taking the log of the Size, the relationship looks more linear
shapiro.test(colony$Minutes)
shapiro.test(colony$Size)
#the normality of size does not look ok
shapiro.test(log(colony$Size))
#now the ariable looks more normal and is ready for regression
summary(model.red <- lm(log(Size)~Minutes, data = colony))
#from the model it is evident that the the model explains 98.9% of the variability of colony size. furthermore, at time zero, the colony size is 33.48
exp(model.red$coefficients[1])
plot(model.red)
#assmptions look like they were satisfied;
shapiro.test(model.red$residuals)
#the test confirms the residuals are normal
plot(model.red$residuals)
#it seems like there is some relation, but the shapiro confirms that they are normal
dwt(model.red)
#DW is not ideal but still okay.
exp(predict(model.red, data.frame(Minutes = 2.5*60), interval = "pred", level = 0.95))
exp(predict(model.red, data.frame(Minutes = 2.5*60), interval = "conf", level = 0.95))
#the prediction is 327.81 with a CI of 183.89 to 584.347 if we take into account the errors of the fitting of the model and the errors not accounted for. and 274 and 392  when taking into account only the errors that arise when fitting the model

#Q2
sales <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Revision for test/sales.csv")
str(sales)
shapiro.test(sales$Sales)
#data is normal
summary(sales.model <- aov(Sales~Shelf.Height, data = sales))
#the one way anova shows that the shelf height have a significance on the saales of dog food - there is sufficient evidence at the 5% significance level to believe that shelf heights affect sales
plot(sales.model, which = c(1,2,5))
#looks like assumptions are adhered to, apart from the homogenity of variance which could be argued
shapiro.test(sales.model$residuals)
#residuals are normal
plot(sales.model$residuals)
#residuals look random
#thus the conditions are satisfied

tuk <- TukeyHSD(sales.model)
tuk
#the tukey posthoc method shows that the sales of Waist level are significantly different from the other two
plot(tuk)
pairwise.t.test(sales$Sales, sales$Shelf.Height, p.adjust.method = "bon")
#the pairwise ttest using the boniferroni correcion concludes the same

#Q3
balance <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Revision for test/balance.csv")
str(balance)
balance$subject <- factor(balance$subject)
balance$condition <- factor(balance$condition)
shapiro.test(balance$time)
#normal data
summary(balance.model <- aov(time~subject+condition, data = balance))
#the model output concludes that there is a difference between means of both subject and condition with a 5% level of significance
plot(balance.model, which = c(1,2,5))
#residuals and normality look satisfied
shapiro.test(balance.model$residuals)
#residuals are normal
plot(balance.model$residuals)
#they look random
interaction.plot(balance$subject, balance$condition, balance$time)
#both have an effect and also they have a joint interaction
interaction.plot(balance$condition, balance$subject, balance$time)
#condition has an effect as well as subject. furthermore, it can be suspected that there is an interaction between the two