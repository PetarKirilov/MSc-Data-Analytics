#Q1
chile <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 8 - Logistics Regression; CrossValidation/Chile.csv")
chile <- na.omit(chile)
chile <- chile[chile$vote %in% c("Y","N"),]
chile$vote <- factor(chile$vote)
levels(chile$vote)
summary(allvoters <- glm(vote~education+region+age+income, data = chile, family = binomial(logit)))
#it seems that all but regionSA are significant in the model. With the others, the relationships are as their coefficients in the regression output - positive, meaning that it increases the chances to vote YES, and viceversa with being negative
exp(cbind(coef(allvoters), confint(allvoters)))
#from this output, only the regionSA includes 1 in its confidence interval, but we already concluded that it is not significant.
summary(malevoters <- glm(vote~education+region+age+income, data = chile, subset = (chile$sex == "M"), family = binomial(logit)))
#again, all but regionSA is significant. All but PS and S education increase the chance to vote YES
exp(cbind(coef(malevoters), confint(malevoters)))
#only regionSA crosses 1 in the confidence interval
summary(femalevoters <- glm(vote~education+region+age+income, data = chile, subset = (chile$sex == "F"), family = binomial(logit)))
#regionS and age is not significant. Only PS and S education and regionSA decreases the chances to vote YES, all others increase the chances
exp(cbind(coef(femalevoters), confint(femalevoters)))
#age, regionN and regionS crosses 1 in the confidence intervals, but it is insignificant from the regression

#Q2
