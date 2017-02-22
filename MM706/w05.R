#problem1
data(mtcars)
cor(mtcars)
v <- cor(mtcars)[1,] #as we want only the first row of the matrix which represents the mpg correlations
#f
v[which.max(v)]
v[which.min(v)]
#g
sort(abs(v), decreasing = TRUE)

#problem 2
#b
model <- lm(mpg~wt, data = mtcars)
#c
plot(mpg~wt, data = mtcars)
abline(model, col = "red")#negative trend
#d
summary(model)
#both are significant, R2 is 75.28%

#Problem 3
d <- read.fwf("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM706 - Programming for Analytics CORE SEM 1 20CR/05 R regression models and decision trees/quiz.txt", width = rep(1,6), stringsAsFactors = FALSE)
dats <- d[1:20, 1:4]
rhs <- as.integer(d[1:20, 6])
install.packages("reshape2")
library(reshape2)
tab <- melt(dats)
tab$rows <- rep(1:nrow(dats), ncol(dats))
tab <- table(tab$rows, tab$value)
mat <- cbind(tab, rhs)
mat <- as.data.frame(mat)
names(mat) <- c("zero", "one", "two", "three", "five", "six", "seven", "eight", "nine", "rhs")
model <- lm(rhs~0+., data = mat) #the dot means all other variables
round(model$coefficients)

#Problem 4
train <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM706 - Programming for Analytics CORE SEM 1 20CR/05 R regression models and decision trees/titanic_train.csv")
testt <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM706 - Programming for Analytics CORE SEM 1 20CR/05 R regression models and decision trees/titanic_test.csv")
train$Pclass <- as.factor(train$Pclass)
testt$Pclass <- as.factor(testt$Pclass)
install.packages("rpart")
library(rpart)
model <- rpart(Survived~Sex+Pclass+Age, data = train)
prediction <- predict(model, testt)
#to do the representation we need to install a couple of packages
install.packages("rattle")
install.packages("rpart.plot")
installed.packages("RColorBrewer")
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(model) #this plots a decision tree of the model


