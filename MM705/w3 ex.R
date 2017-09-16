library(Hmisc)
library(car)
library(reshape2)
library(Hmisc)
#Lecture example
radon <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/radon.csv")
radon <- melt(radon)
#We use the melt function to represent the tests in column view
names(radon) <- c("Device", "Radiation")
shapiro.test(radon$Radiation)
#according to the shapiro wilk test, the data is not normal
summary(radon.model <- aov(Radiation~Device, data = radon))
#the anova concludes that the means are different
plot(radon.model, which = c(1,2,5))
#assumptions look plausible
shapiro.test(radon.model$residuals)
#resids test normal
bartlett.test(Radiation~Device, data = radon)
#the bartlett test concludes that the variance is not equal, thus the anova cannot be used ,unless the model is modified
#from the plots, it is evident that the Membrane set is different so we will remove it
summary(radon.model.nomemb <- aov(Radiation~Device, data = radon[radon$Device != "Membrane", ]))
#similar results but lets check the assumptions
plot(radon.model.nomemb, which = c(1,2,5))
#looks better
shapiro.test(radon.model.nomemb$residuals)
#residuals test normal
bartlett.test(Radiation~Device, data = radon[radon$Device != "Membrane", ])
#the bartlett test confirms that the variance is equal

#Randomized block design
tensile <- spss.get("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/tensile.sav")
tensile$bolt <- factor(tensile$bolt)
tensile$chemical <- factor(tensile$chemical)
shapiro.test(tensile$strength)
#normal data
class(tensile$strength) <- "numeric"
summary(model.1.tensile <- aov(strength~chemical, data = tensile))
#p value quite large, need to check with the bolt
summary(model.tensile <- aov(strength~chemical+bolt, data = tensile))
#now it seems that only the bolt is significant
plot(model.tensile, which = c(1,2,5))
shapiro.test(model.tensile$residuals)
#normal residuals
summary(model.E.tensile <- aov(strength~chemical+Error(bolt/chemical),data=tensile))
#this includes it in the error term
shapiro.test(model.E.tensile$`bolt:chemical`$residuals)
#the residuals between the interaction of bolt and chemical is normal
friedman.test(strength~chemical|bolt, data = tensile)

#Q1
billiard <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/billiard.csv")
billiard$Batch <- factor(billiard$Batch)
shapiro.test(billiard$Elastic)
#data is normal
ks.test(billiard$Elastic, pnorm, mean(billiard$Elastic), sd(billiard$Elastic))
summary(billiard.model <- aov(billiard$Elastic~billiard$Additive+billiard$Batch))
plot(billiard.model, which = c(1,2,5))
#the assumptions do not look satisfied
shapiro.test(billiard.model$residuals)
#residuals are borderline normal
bartlett.test(Elastic~Additive, data = billiard)
#the bartlett test for homogenity of variance concludes that there is no difference in variances
friedman.test(Elastic~Additive|Batch, data = billiard)
#the friedman test concludes that the means are different, meaning that the additives affext the elasticity of the balls

#Q2
no <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/NO.csv")
no$Driver <- factor(no$Driver)
no$Car <- factor(no$Car)
shapiro.test(no$Emission.rate)
#according to the shapiro wilk test the data is not normal
summary(no.model <- aov(Emission.rate~Driver+Car, data = no))
#anova output identifies that the driver affects the performance
plot(no.model, which = c(1,2,5))
#the model does not satisfy the asusmptions of the ANOVA
bartlett.test(Emission.rate~Driver, data = no)
#the test confirms that the variances are homoscedastic
shapiro.test(no.model$residuals)
#the shapiro wilk test on residuals concludes that they are normally distributed
friedman.test(Emission.rate~Driver|Car, data = no)
#the friedman rank test concludes the same as the ANOVA - some drivers get lower rates than others

#Rabbits example
rabbit <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/rabbits.csv")
rabbit$Enclosure <- factor(rabbit$Enclosure)
rabbit$Distance <- factor(rabbit$Distance)
shapiro.test(rabbit$Crop)
#normal data
summary(rabbit.model <- aov(Crop~Distance+Enclosure, data = rabbit))
#the Enclosure looks like has a different mean
plot(rabbit.model, which = c(1,2,5))
#assumptions hold
bartlett.test(Crop~Enclosure, data = rabbit)
#variance is equal
shapiro.test(rabbit.model$residuals)
#normal residuals
interaction.plot(rabbit$Enclosure, rabbit$Distance, rabbit$Crop)
#there seems to be a large effect(the crop increases with enclosure), which is confirmed by the ANOVA
interaction.plot(rabbit$Distance, rabbit$Enclosure, rabbit$Crop)
#there seems to be a small effect from the distance, which is confirming the ANOVA
tuk <- TukeyHSD(rabbit.model)
plot(tuk)


#Q6
virus <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/virus.csv")
str(virus)
virus$Time <- factor(virus$Time)
interaction.plot(virus$Time, virus$Medium, virus$Radius )
#it confirms that with time, the radius increases, however, when the time reaches 15, B and C start to decrease.
interaction.plot(virus$Medium, virus$Time, virus$Radius)
#it seems like C and 9 and 15 TIME interact - they increase
shapiro.test(virus$Radius)
#data tests normal
summary(virus.model <- aov(Radius~Medium*Time, data = virus))
#it seems like all the variables and ineractions are signiciant
plot(virus.model, which = c(1,2,5))
#it seems that the assumptions hold
shapiro.test(virus.model$residuals)
#resids test normal
bartlett.test(Radius~Time, data = virus)
#the homogenity of variance with time shows that the variance is not equal
bartlett.test(Radius~Medium, data = virus)
#variance is equal
#The fastest growth of the virus occurs for time 15 and C culture media

#Q7
ai <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/Exercise7ai.csv")
ai$A <- factor(ai$A)
ai$B <- factor(ai$B)
shapiro.test(ai$Response)
#normal
summary(ai.model <- aov(Response~A*B, data = ai))
#only A seems to differ
shapiro.test(ai.model$residuals)
#normal
plot(ai.model, which = c(1,2,5))
#variance does not look normal
bartlett.test(Response~A, data = ai)
#it is bordeline, as the p-value is 0.05818, but still variances are equal
plot(ai.model$residuals)
#YES
interaction.plot(ai$A, ai$B, ai$Response)
interaction.plot(ai$B, ai$A, ai$Response)

aii <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/Exercise7aii.csv")
aii$A <- factor(aii$A)
aii$B <- factor(aii$B)
shapiro.test(aii$Response)
#normal
summary(aii.model <- aov(Response~A*B, data = aii))
#both A and B seem to differ
shapiro.test(aii.model$residuals)
#normal
plot(aii.model, which = c(1,2,5))
#looks normal
bartlett.test(Response~A, data = aii)
#variances are equal
plot(aii.model$residuals)
#a bit of a problem with the residuals, as there is a trend
#NO
interaction.plot(aii$A, aii$B, aii$Response)
interaction.plot(aii$B, aii$A, aii$Response)

aiii <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/Exercise7aiii.csv")
aiii$A <- factor(aiii$A)
aiii$B <- factor(aiii$B)
shapiro.test(aiii$Response)
#normal
summary(aiii.model <- aov(Response~A*B, data = aiii))
#both A and B seem to differ
shapiro.test(aiii.model$residuals)
#normal
plot(aiii.model, which = c(1,2,5))
#looks normal
bartlett.test(Response~A, data = aiii)
#variances are equal
plot(aiii.model$residuals)
#residuals look random
#MAYBE
interaction.plot(aiii$A, aiii$B, aiii$Response)
interaction.plot(aiii$B, aiii$A, aiii$Response)

biii <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/Exercise7biii.csv")
biii$A <- factor(biii$A)
biii$B <- factor(biii$B)
shapiro.test(biii$Response)
#normal
summary(biii.model <- aov(Response~A*B, data = biii))
#only interaction seems to differ
shapiro.test(biii.model$residuals)
#not normal
plot(biii.model, which = c(1,2,5))
#looks normal
bartlett.test(Response~A, data = biii)
#variances are not equal
plot(biii.model$residuals)
#there is some pattern with the residuals 
#NO
interaction.plot(biii$A, biii$B, biii$Response)
interaction.plot(biii$B, biii$A, biii$Response)

bii <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/Exercise7bii.csv")
bii$A <- factor(bii$A)
bii$B <- factor(bii$B)
shapiro.test(bii$Response)
#not normal
summary(bii.model <- aov(Response~A*B, data = bii))
#only interaction seems to differ
shapiro.test(bii.model$residuals)
#normal
plot(bii.model, which = c(1,2,5))
#looks normal
bartlett.test(Response~A, data = bii)
#variances are equal
plot(bii.model$residuals)
#residuals look random
#YES
interaction.plot(bii$A, bii$B, bii$Response)
interaction.plot(bii$B, bii$A, bii$Response)

bi <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/Exercise7bi.csv")
bi$A <- factor(bi$A)
bi$B <- factor(bi$B)
shapiro.test(bi$Response)
#not normal
summary(bi.model <- aov(Response~A*B, data = bi))
#only interaction seems to differ
shapiro.test(bi.model$residuals)
#normal
plot(bi.model, which = c(1,2,5))
#looks normal
bartlett.test(Response~A, data = bi)
#variances are not equal
plot(bi.model$residuals)
#there is no pattern with the residuals 
#NO
interaction.plot(bi$A, bi$B, bi$Response)
interaction.plot(bi$B, bi$A, bi$Response)
