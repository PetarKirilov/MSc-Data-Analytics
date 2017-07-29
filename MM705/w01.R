library(Hmisc)
setwd("C:/Users/pkk11/Downloads")
test <- spss.get("pulse.sav")
t.test(test$pulse1,mu=75)
#we cannot reject the null as the pvalue is more than 5%, thus the mean is equal to 75
wilcox.test(test$pulse1,mu=75)
#we cannot reject the null as the pvalue is more than 5%, thus the mean is equal to 75
install.packages("BSDA")
require(BSDA)
SIGN.test(test$pulse1, md=75)
#we can reject the null as the pvalue is less than 5%, thus median is not equal to 75
#TESTING for females
t.test(test$pulse1[test$sex == "female"],mu=75)
#we cannot reject the null as the pvalue is more than 5%, with CI of 72.87 and 80.85
wilcox.test(test$pulse1[test$sex == "female"], mu=75)
#we cannot reject the null as the pvalue is more than 5%
SIGN.test(test$pulse1[test$sex == "female"],md=75)
#we cannot reject the null as the pvalue is more than 5%
t.test(test$pulse1[test$sex == "female"], mu=72, alternative = "g")
#we can strongly reject the null as the pvaule is less than 5%
wilcox.test(test$pulse1[test$sex == "female"], mu=72, alternative = "g")
#we can reject the null as the pvalue is less than 5%
SIGN.test(test$pulse1[test$sex == "female"], md=72, alternative = "g")
#we can reject the null as the pvalue is less than 5%, with CI 72 and infinity
#TEST FOR NORMALITY
shapiro.test(test$pulse1[test$sex == "female"])
#pvalue is high thus we cannot reject the null that the data is non-parametric
hist(test$pulse1[test$sex == "female"])
qqnorm(test$pulse1[test$sex == "female"])
qqline(c(test$pulse1[test$sex == "female"]))
