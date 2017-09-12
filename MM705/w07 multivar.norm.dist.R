library(forecast)
#Q2
tr <- trees
plot(tr)
summary(tr.model <- lm(Volume~Girth+Height, data = tr))
CV(tr.model)
summary(tr.model1 <- lm(log(Volume)~Girth+Height, data = tr))
CV(tr.model1)
plot(tr.model1)
#by taking the log transformation of Volume, the R2 adj has increased indicating a better fit of the model
summary(tr.model2 <- lm(Volume~log(Girth)+log(Height), data = tr))
plot(tr.model2)
#this is looking worse with worsened R2
summary(tr.model3 <- lm(Volume~Girth, data = tr))
plot(tr.model3)
CV(tr.model3)
#this reduced model gives good R2
summary(tr.model3log <- lm(log(Volume)~Girth, data = tr))
CV(tr.model3log)
plot(tr.model3log)
#if we take the full model with the logged volume as the best one, the confidence intervals are:
confint(tr.model1)
#these are the conf intervals of the values

#Q3
library(fpp)
tgas <- texasgas
plot(tgas)
summary(tgas.model <- lm(consumption~log(price), data = tgas))
plot(tgas.model)
#looks like the assumptions are met
summary(tgas.model.less60 <- lm(consumption~price, data = tgas, subset = (tgas$price<=60)))
CV(tgas.model.less60)
plot(tgas.model.less60$residuals)
summary(tgas.model.more60 <- lm(consumption~price, data = tgas, subset = (tgas$price>60)))
CV(tgas.model.more60)
plot(tgas.model.more60$residuals)
#lower AIC but R2 is lower as well
summary(tgas.model.sqdum <- lm(consumption~price+I(price^2), data = tgas))
CV(tgas.model.sqdum)
plot(tgas.model.sqdum$residuals)
#judging by the Rsqr, this is the best model, but it is not comparable to the other two as they are looking at different data

#Q4
