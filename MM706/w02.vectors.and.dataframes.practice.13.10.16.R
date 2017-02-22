#problem 1
str(d)
f <- d$price >= 15000
sum(f)
d.fair <- d[d$cut == "Fair",]
d.good <- d[d$cut == "Good",]
d.ideal <- d[d$cut == "Ideal",]
d.vgood <- d[d$cut == "Very Good",]
d.premium <- d[d$cut == "Premium",]
max(d.premium$price)
min(d.premium$price)
median(d.premium$price)

#problem 2
titanic <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM706 - Programming for Analytics CORE SEM 1 20CR/02 R data frames/titanic.csv")
str(titanic)
survived <- sum(titanic$Survived)
total <- nrow(titanic)
not.surv <- total - survived
titanic.male <- titanic[titanic$Sex == "male",]
survived.male <- sum(titanic.male$Survived)
titanic.female <- titanic[titanic$Sex == "female",]
survived.female <- sum(titanic.female$Survived)
survived.male/nrow(titanic.male)
survived.female/nrow(titanic.female)
titanic.child <- na.omit(titanic[titanic$Age < 18,])
survived.child <- sum(titanic.child$Survived)
survived.child/nrow(titanic.child)
class1 <- titanic[titanic$Pclass == 1,]
class2 <- titanic[titanic$Pclass == 2,]
class3 <- titanic[titanic$Pclass == 3,]
survived.class1 <- sum(class1$Survived)
survived.class2 <- sum(class2$Survived)
survived.class3 <- sum(class3$Survived)
survived.class1/nrow(class1)
survived.class2/nrow(class2)
survived.class3/nrow(class3)

#problem 3
boys <- data.frame(id = 1:8, grade = as.integer(100*runif(8)), gender = rep("boy", 8))
girls <- data.frame(id = 1:10, grade = as.integer(100*runif(10)), gender = rep("girl", 10))
df <- as.data.frame(rbind(boys, girls))

#problem 4
boys <- data.frame(id = 1:8, grade = as.integer(100*runif(8)))
girls <- data.frame(Number = 1:10, Mark = as.integer(100*runif(10)))
names(boys) <- c("id", "grade")
names(girls) <- c("id", "grade")
girls$gender <- rep("girl", 10)
boys$gender <- rep("boy", 8)
rbind(boys, girls)

#problem 5
