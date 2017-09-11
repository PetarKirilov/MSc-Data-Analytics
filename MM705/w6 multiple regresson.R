#w6 Multiple regression exercises
pairs(airquality)
Airquality <- airquality[complete.cases(airquality),]
ozone.model <- lm(Ozone~Solar.R+Wind+Temp, data = Airquality)
summary(ozone.model)
install.packages("forecast")
library(forecast)
plot(Airquality$Ozone, ozone.model$fitted.values)
CV(ozone.model)


#Q1
tr <- trees
plot(tr)
summary(tr.model <- lm(Volume~Girth+Height, data = tr))
CV(tr.model)
summary(tr.model1 <- lm(log(Volume)~Girth+Height, data = tr))
CV(tr.model1)
#by taking the log transformation of Volume, the R2 adj has increased and the AIC and BIC have decreased indicating a better model


#Q3
air <- airquality
str(air)
air$Month <- factor(air$Month)
air$Day <- factor(air$Day)
air75 <- 