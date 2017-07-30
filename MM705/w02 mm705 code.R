library(Hmisc)
#ex1
amino <- spss.get("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 2 - One-way ANOVA/amino.sav")
hist(amino$MDA)
levels(amino$AMINO)
pairwise.t.test(amino$MDA, amino$AMINO, p.adjust.method = "none", pool.sd = FALSE)
plot(unclass(amino$AMINO), amino$MDA)
boxplot(amino$MDA~amino$AMINO)
model1 <- aov(MDA~AMINO, data = amino)
summary(model1)
#checking the assumptions
plot(model1$fitted.values,model1$residuals,main="Residuals vs. Fitted")
plot(1:length(model1$model$AMINO),model1$residuals,main="Residuals vs. Order")
#the above shows that the residuals are on a straight lines, meaning it holds the normality assumption and the points are scattered
qqnorm(model1$residuals)
qqline(c(model1$residuals))
kruskal.test(MDA~AMINO, data = amino)

#ex2
tablet <- spss.get("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 1 - One and Two sample tests/tablets.sav")
tablet$MS<-factor(tablet$MS)
tablet$COMPRESS<-factor(tablet$COMPRESS)
modelT<-aov(TIME~COMPRESS,data=tablet,subset=(tablet$MS==0.5))
summary(modelT)
plot(modelT$fitted.values,modelT$residuals,main="Residuals vs. Fitted")
plot(1:length(modelT$model$COMPRESS),modelT$residuals,main="Residuals vs. Order")
qqnorm(modelT$residuals)
qqline(c(modelT$residuals))
#normality assumptions do hold
pairwise.t.test(tablet$TIME[tablet$MS==0.5], tablet$COMPRESS[tablet$MS==0.5], p.adjust.method = "none", pool.sd = FALSE)
kruskal.test(TIME~COMPRESS, data = tablet, subset = (tablet$MS==0.5))
#1%
modelT1<-aov(TIME~COMPRESS,data=tablet,subset=(tablet$MS==1))
summary(modelT1)
plot(modelT1$fitted.values,modelT1$residuals,main="Residuals vs. Fitted")
plot(1:length(modelT1$model$COMPRESS),modelT1$residuals,main="Residuals vs. Order")
qqnorm(modelT1$residuals)
qqline(c(modelT1$residuals))
#normality of assumptions do hold
pairwise.t.test(tablet$TIME[tablet$MS==1], tablet$COMPRESS[tablet$MS==1], p.adjust.method = "none", pool.sd = FALSE)
kruskal.test(TIME~COMPRESS, data = tablet, subset = (tablet$MS==1))

#included in errors
modelTnot<-aov(TIME~COMPRESS,data=tablet)
summary(modelTnot)
plot(modelTnot$fitted.values,modelTnot$residuals,main="Residuals vs. Fitted")
plot(1:length(modelTnot$model$COMPRESS),modelTnot$residuals,main="Residuals vs. Order")
qqnorm(modelTnot$residuals)
qqline(c(modelTnot$residuals))
pairwise.t.test(tablet$TIME, tablet$COMPRESS, p.adjust.method = "none", pool.sd = FALSE)
kruskal.test(TIME~COMPRESS, data = tablet)
