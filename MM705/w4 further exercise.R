steel <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 4 - Factorial Designs Continued; Multiple Comparisons/steel.csv")
str(steel)
steel$A <- factor(steel$A)
steel$B <- factor(steel$B)
steel$C <- factor(steel$C)
interaction.plot(steel$A, steel$B, steel$Length)
shapiro.test(steel$Length)
#normal
summary(steel.model <- aov(Length~A*B*C, data = steel))
summary(steel.model.1 <- aov(Length~A*B+C+A:C+B:C, data = steel))
summary(steel.model.2 <- aov(Length~A*B+C+A:C, data = steel))
summary(steel.model.3 <- aov(Length~A+B+C+A:C, data = steel))
summary(steel.model.4 <- aov(Length~B+C+A:C, data = steel))
#only B and C are significant and C:A interaction is almost significant but is outside the 5% treshold
shapiro.test(steel.model.4$residuals)
#good
plot(steel.model.4$residuals)
#good, random
plot(steel.model.4, which = c(1,2,5))
#good model


#Q2
blood <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 4 - Factorial Designs Continued; Multiple Comparisons/bloodpressure.csv")
str(blood)
shapiro.test(blood$Bloodpressure)
#normal
summary(blood.model <- aov(Bloodpressure~Diet*Drug*Biofeed, data = blood))
summary(blood.model.1 <- aov(Bloodpressure~Diet*Drug*Biofeed-Diet:Biofeed, data = blood))
summary(blood.model.2 <- aov(Bloodpressure~Diet+Drug+Biofeed+Diet:Drug, data = blood))
#all are signif
plot(blood.model.2$residuals)
#ranodm
shapiro.test(blood.model.2$residuals)
#not normal.
plot(blood.model.2, which = c(1,2,5))
#seems reasonable

#Q4
stress <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 4 - Factorial Designs Continued; Multiple Comparisons/profstress.csv")
str(stress)
library(reshape2)
stress <- melt(stress)
str(stress)
shapiro.test(stress$value)
#normal
summary(stress.model <- aov(value~variable, data = stress))
#significant
plot(stress.model$residuals)
#random
shapiro.test(stress.model$residuals)
#good
plot(stress.model, which = c(1,2,5))
#looks okay
tuk1 <- TukeyHSD(stress.model)
tuk1
plot(tuk1)
#systems analysts, lawyers, layers have significant differences
pairwise.t.test(stress$value, stress$variable, p.adjust.method = "bon")
#similar coclusions


#Q5
yield <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 4 - Factorial Designs Continued; Multiple Comparisons/yield.csv")
str(yield)
