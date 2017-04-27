library(Hmisc)
#ex1
billiard <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/billiard.csv")
names(billiard)
friedman.test(Elastic~Additive|Batch, data = billiard)
summary(elastic_model2 <- aov(Elastic~Additive+Batch, data = billiard))
plot(elastic_model2, which = c(1,2,5))
#friedman is more appropriate as the data is not normal
shapiro.test(billiard$Elastic)
#the above shapirowilk test contradicts showing that the data is normal
#ex2
no <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/NO.csv")
names(no)
summary(no_model <- aov(Emission.rate~Driver+Car, data = no))
plot(no_model, which = c(1,2,5))
friedman.test(Emission.rate~Driver|Car, data = no)
#friedman is more appropriate as the data is not normal

#ex3
rabbit <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/rabbits.csv")
names(rabbit)
summary(rabbit_model <- aov(Crop~Distance+Enclosure, data = rabbit))
#1
interaction.plot(rabbit$Distance, rabbit$Enclosure, rabbit$Crop, trace.label = "Enclosure", xlab = "Distance", ylab = "Crop", col = c("red", "blue"))
#from the above interaction plot, because the lines are paralel, it can be concluded that both have an effect
interaction.plot(rabbit$Enclosure, rabbit$Distance, rabbit$Crop,  trace.label = "Distance", xlab = "Enclosure", ylab = "Crop", col = c("red", "blue"))
#this confirms the below conclusion that both have an effect
#2
summary(rabbit_model <- aov(Crop~Distance+Enclosure, data = rabbit))
#3
plot(rabbit_model, which = c(1,2,5))
#ANOVA assumptions hold
#ex4
tablets <- spss.get("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 1 - One and Two sample tests/tablets.sav")
tablets$MS<-factor(tablets$MS)
tablets$COMPRESS<-factor(tablets$COMPRESS)
interaction.plot(tablets$COMPRESS, tablets$MS, tablets$TIME, col = c("red", "blue"))
interaction.plot(tablets$MS, tablets$COMPRESS, tablets$TIME, col = c("red", "blue", "green"))
#paralel lines, thus both have an effect
summary(tablets_model <- aov(TIME~COMPRESS+MS, data = tablets))
#ANOVA confirms that both have an effect
plot(tablets_model, which = c(1,2,5))
#ANOVA assumptions seem to hold

#ex5
afric <- spss.get("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/vision.sav")
interaction.plot(afric$SEED, afric$FLOWER, afric$VISION)
interaction.plot(afric$FLOWER, afric$SEED, afric$VISION)

summary(afric_model <- aov(VISION~FLOWER+SEED, data = afric))
#ANOVA shows that 
plot(afric_model, which = c(1,2,5))



starlings <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 3 - Two-Way ANOVA Lecture/starlings.csv")
class(starlings$Gender)
