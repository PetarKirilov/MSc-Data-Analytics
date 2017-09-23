library(car)
library(datasets)
library(stats)
library(graphics)
#Q1
davis <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 5 - Regression/Davis.csv")
plot(davis$weight, davis$height)
#outlier at 160
davis <- davis[!(davis$weight==166),]


#Q2
summary(car::Anscombe)
plot(car::Anscombe)
ansombe <- car::Anscombe
ff <- y ~ x
mods <- setNames(as.list(1:4), paste0("lm", 1:4))
for(i in 1:4) {
  ff[2:3] <- lapply(paste0(c("y","x"), i), as.name)
  ## or   ff[[2]] <- as.name(paste0("y", i))
  ##      ff[[3]] <- as.name(paste0("x", i))
  mods[[i]] <- lmi <- lm(ff, data = anscombe)
  print(anova(lmi))
}
sapply(mods, coef)
lapply(mods, function(fm) coef(summary(fm)))
op <- par(mfrow = c(2, 2), mar = 0.1+c(4,4,1,1), oma =  c(0, 0, 2, 0))
for(i in 1:4) {
  ff[2:3] <- lapply(paste0(c("y","x"), i), as.name)
  plot(ff, data = anscombe, col = "red", pch = 21, bg = "orange", cex = 1.2,
       xlim = c(3, 19), ylim = c(3, 13))
  abline(mods[[i]], col = "blue")
}
mtext("Anscombe's 4 Regression data sets", outer = TRUE, cex = 1.5)
par(op)


#Q3
econsumption <- data.frame(Day = c(1:12), Mwh = c(16.3,	16.8,	15.5,	18.2,	15.2,	17.5,	19.8,	19.0,	17.5,	16.0,	19.6,	18.0), temp = c(29.3,	21.7,	23.7,	10.4,	29.7,	11.9,	9.0,	23.4,	17.8,	30.0,	8.6,	11.8))
plot(econsumption$Mwh~econsumption$temp)
#less electricity used for heating
reg <- lm(econsumption$Mwh~econsumption$temp)
plot(reg$residuals)
confint(reg)

#Q4
plot(cars)
abline(lm(cars$dist~cars$speed))
summary(cars.model.n <- lm(dist~speed, data = cars))
plot(log(cars$speed), log(cars$dist))
abline(lm(log(cars$dist)~log(cars$speed)))
summary(cars.model <- lm(log(dist)~log(speed), data = cars))
plot(cars.model)
shapiro.test(cars.model$residuals)
exp(cars.model$coefficients)


#Q5
orange.model <- lm(circumference~age, data = Orange)
plot(Orange$circumference~ Orange$age)
abline(orange.model)
summary(orange.model)
plot(orange.model)
shapiro.test(orange.model$residuals)
dwt(orange.model)
orange.log.model <- lm(log(circumference)~log(age), data = Orange)
summary(orange.log.model)
shapiro.test(orange.log.model$residuals)
dwt(orange.log.model)
plot(orange.log.model)
plot(log(Orange$circumference)~ log(Orange$age))
abline(orange.log.model)
predict(orange.model, data.frame(age = 750), interval = "conf", level = 0.95)
exp(predict(orange.log.model, data.frame(age = 750), interval = "conf", level = 0.95))

#Q6
island <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 5 - Regression/islands.csv")
plot(island$No.species~island$Area)
island.model <- lm(island$No.species~island$Area)
abline(island.model)
island.model.log <- lm(log(No.species)~log(Area), data = island)
island.sqrt <- lm((island$No.species)~sqrt(island$Area))
plot(log(island$No.species)~log(island$Area))
abline(island.model.log)
exp(predict(island.model.log, data.frame("Area" = 1200), interval = "confidence", level = 0.95))

colony <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Revision for test/colony2.csv")
plot(colony)
plot(colony$Minutes, log(colony$Size))
abline(log(colony$Size)~colony$Minutes)
colony.lm <- lm(log(Size)~Minutes, data = colony)
exp(predict(colony.lm, data.frame(Minutes=2.5*60), interval = "conf"))
