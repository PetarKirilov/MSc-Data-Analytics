#EX1
davis <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 5 - Regression/Davis.csv")
plot(davis)
plot(davis$weight, davis$repwt)
#evident that there is an outlier with value more than 160 in weight, so we need to remove it
max(davis$weight)
#this is the value
davis <- davis[!(davis$weight==166),]
#the above removes the outlier
plot(davis$weight, davis$repwt)
abline(lm(davis$weight ~ davis$repwt))
#a scatter plot of the data and the regression line
summary(davis.weight.model <- lm(davis$weight ~ davis$repwt))
#the R^2 is 97%, both coefficients are with a p-value of less than 5%; the intercept coefficient (beta0) is 2.7389 and the beta1 coefficient is 0.95832 which confirms their positive correlation
#the regression equation is y=2.7389+0.95832x where y is the weight and x is the reported weight

plot(davis$height, davis$repht)
#there seems like there is a linear relation
#lets plot the regression line
abline(lm(davis$height~davis$repht))
summary(davis.height.model <- lm(davis$height~davis$repht))
#R^2 is 95%; both coefficient are significant with a p-value of less than 5%; the intercept coefficient is 12.76117 and the beta1 coefficient is 0.93653 which confirms the positive correlation
#the regression equation is y=12.76117+0.93653x where y is the height and x us the reported height

plot(davis$weight, davis$height)
abline(lm(davis$weight~davis$height))
summary(davis.heightweight.model <- lm(davis$weight~davis$height))
#the R^2 is quite low (59.5%) meaning that 59 percent of weight is explained by height; both coefficients are significant; the intercept coefficient is -131.66997 and the beta1 coefficient is 1.15442 confirming the positive correlation
#the regression equation is y=-131.66997+1.15442x

plot(davis$weight[davis$sex == "M"], davis$repwt[davis$sex == "M"])
abline(lm(davis$weight[davis$sex == "M"]~davis$repwt[davis$sex == "M"]))
#seems like there is a linear relation
summary(davis.males.weight.model <- lm(davis$weight[davis$sex == "M"]~davis$repwt[davis$sex == "M"]))
#R^2 is 95.91%, with the beta1 coefficient being significant  and the intercept

plot(davis$weight[davis$sex == "F"], davis$repwt[davis$sex == "F"])
abline(lm(davis$weight[davis$sex == "F"]~davis$repwt[davis$sex == "F"]))
#seems like there is a linear relation
summary(davis.females.weight.model <- lm(davis$weight[davis$sex == "F"]~davis$repwt[davis$sex == "F"]))

#in order to plot both on the scatter plot, we use the below code
plot(davis$weight, davis$repwt, col = ifelse(davis$sex == "M", "red", "blue"), cex = 1.5)
abline(lm(davis$weight[davis$sex == "M"]~davis$repwt[davis$sex == "M"]), col = "dark red", cex = 1.5)
abline(lm(davis$weight[davis$sex == "F"]~davis$repwt[davis$sex == "F"]), col = "dark blue", cex = 1.5)
legend("topleft", c("Male", "Female"), col = c("red", "blue"), pch = c(1,1), cex=1.5)
legend("bottomright", c("Regression Male", "Regression Female"), col = c("dark red", "dark blue"), lty = c(1), cex = 1.5)
#NEED TO FIX ABOVE LEGEND
confint(davis.females.weight.model, level = 0.95)
confint(davis.males.weight.model, level = 0.95)
