#Q6 W3
virus <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/virus.csv")
str(virus)
virus$Time <- factor(virus$Time)
virus$Medium <- factor(virus$Medium)
shapiro.test(virus$Radius)
#normal
summary(virus.model <- aov(Radius~Time*Medium, data = virus))
#the model shows that Time, Medium and the interaction between are significant and cause an effect
shapiro.test(virus.model$residuals)
#normal resids
plot(virus.model, which = c(1,2,5))
#looks alright
interaction.plot(virus$Time, virus$Medium, virus$Radius)
#interaction plot confirms the anova
interaction.plot(virus$Medium, virus$Time, virus$Radius)
#I would recommend sticking to medium C and stop at time 15.
library(Hmisc)
afric <- spss.get("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/vision.sav")
str(afric)
afric$FLOWER <- factor(afric$FLOWER)
afric$SEED <- factor(afric$SEED)
afric$VISION <- as.numeric(afric$VISION)
shapiro.test(afric$VISION)
#not normal data, thus we cannot use ANOVA
print(afric.model <- kruskal.test(VISION~SEED*VISION, data = afric))
#the test confirms that there is a significant difference in vision either due to seed vision or both
#however, the task wants two way anova:
summary(afric.anova.mod <- aov(VISION~SEED*FLOWER, data = afric))
#it states that all are significant and the means are not equal
plot(afric.anova.mod$residuals)
#random resids
shapiro.test(afric.anova.mod$residuals)
#normal
plot(afric.anova.mod, which = c(1,2,5))
#residuals are abit strange, otherwise normal
