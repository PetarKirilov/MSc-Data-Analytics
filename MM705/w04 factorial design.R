library(reshape2)
#ex1
steel <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 4 - Factorial Designs Continued; Multiple Comparisons/steel.csv")
steel$A <- factor(steel$A, ordered = TRUE)
steel$B <- factor(steel$B, ordered = TRUE)
steel$C <- factor(steel$C, ordered = TRUE)
acast(steel,A~C, length, value.var = "Length")
acast(steel,A~B, length, value.var = "Length")
acast(steel, B~C, length, value.var = "Length")
interaction.plot(steel$A, steel$B, steel$Length)
#lines are not coincided, small interaction, indicating effect of steel$B
interaction.plot(steel$B, steel$A, steel$Length)
#seems like steel$B has an effect as the lines are paralel and going down, which confirms the above
interaction.plot(steel$A, steel$C, steel$Length)
#as the lines are not coinciding and some of them cross, there might be some effect of both A and C
interaction.plot(steel$C, steel$A, steel$Length)
#there might be some effect
interaction.plot(steel$B, steel$C, steel$Length)
#lines are paralel, seems like steel$B has an effect as the lines are paralel and sloping down
interaction.plot(steel$C, steel$B, steel$Length)
#lines do not cross, but there might be some effect as they vary
summary(steel.model <- aov(Length~A*B*C, data = steel))
#it seems like only B, C and A:C is significant
plot(steel.model, which = c(1,2,5))
#ANOVA assumptions hold
summary(steel.model.reduced1 <- aov(Length~B+C*A, data = steel))
#again, just B, C and A:C is significant
plot(steel.model.reduced1, which = c(1,2,5))
summary(steel.model.reduced <- aov(Length~B, data = steel))
#excluding the insignificant factors, the conclusion from above is confirmed
plot(steel.model.reduced, which = c(1,2,5))
#ANOVA assumptions seem to hold

#ex2
bpres <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 4 - Factorial Designs Continued; Multiple Comparisons/bloodpressure.csv")
bpres$Diet <- factor(bpres$Diet, ordered = TRUE)
bpres$Drug <- factor(bpres$Drug, ordered = TRUE)
bpres$Biofeed <- factor(bpres$Biofeed, ordered = TRUE)
acast(bpres, Diet~Drug, length, value.var = "Bloodpressure")
acast(bpres, Diet~Biofeed, length, value.var = "Bloodpressure")
acast(bpres, Biofeed~Drug, length, value.var = "Bloodpressure")
interaction.plot(bpres$Diet, bpres$Drug, bpres$Bloodpressure)
interaction.plot(bpres$Drug, bpres$Diet, bpres$Bloodpressure)
interaction.plot(bpres$Diet, bpres$Biofeed, bpres$Bloodpressure)
interaction.plot(bpres$Biofeed, bpres$Diet, bpres$Bloodpressure)
interaction.plot(bpres$Drug, bpres$Biofeed, bpres$Bloodpressure)
interaction.plot(bpres$Biofeed, bpres$Drug, bpres$Bloodpressure)
summary(bpres.model <- aov(Bloodpressure~Diet*Drug*Biofeed, data = bpres))
plot(bpres.model, which = c(1,2,5))
#Assumptions hold
summary(bpres.reduced.model1 <- aov(Bloodpressure~Diet+Drug+Biofeed+Diet:Drug+Diet:Drug:Biofeed+Diet:Biofeed, data = bpres))
#from the above output we will remove the interaction between Diet and Biofeed
summary(bpres.reduced.model2 <- aov(Bloodpressure~Diet+Drug+Biofeed+Diet:Drug+Diet:Drug:Biofeed, data = bpres))
#from the above output it can be concluded that the interaction between the three can be removed
summary(bpres.reduced.model <- aov(Bloodpressure~Diet+Drug+Biofeed+Diet:Drug, data = bpres))
plot(bpres.reduced.model, which = c(1,2,5))
#asumptions hold

#ex4
prstress <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 4 - Factorial Designs Continued; Multiple Comparisons/profstress.csv")
prstress$Estate.Agents <- factor(prstress$Estate.Agents, ordered = TRUE)
prstress$Architects <- factor(prstress$Architects, ordered = TRUE)
prstress$Stock.Brokers <- factor(prstress$Stock.Brokers, ordered = TRUE)
prstress$Lawyers <- factor(prstress$Lawyers, ordered = TRUE)
prstress$Systems.Analysts <- factor(prstress$Systems.Analysts, ordered = TRUE)
# ne raboti summary(prstress.model <- aov(Estate.Agents*Architects*Stock.Brokers*Lawyers*Systems.Analysts, data = prstress))

