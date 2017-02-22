#Problem 1
titanic <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM706 - Programming for Analytics CORE SEM 1 20CR/02 R data frames/titanic.csv")
#a,b,c
str(titanic)
#d
prop.table(table(titanic$Survived))*100
#e
mf <- table(titanic$Survived, titanic$Sex)
prop.table(mf,2)*100
#f
prop.table(table(titanic$Survived, titanic$Age < 18),2)*100
#g
prop.table(table(titanic$Survived, titanic$Pclass),2)*100

#Problem 2
#c
asd <- prop.table(apply(Titanic, c(1,2,4), sum),c(1,2)) #for the margins of the prop.table function we use c(1,2) as the 
#table is 3 dimentional and we want to preserve 2 dimentions of the 3 dim table. So we choose them as margins of the 
#prop.table function.
asd[,,2] #next we need to show just the Class and Sex of the 3d table who survived so we can reference to them using
#the square brackets notation to reference to the third dimentions and ask R to shows us just the survived passengers.
#a
asdA <- prop.table(apply(Titanic, c(2:4), sum),c(1,2))
asdA[,,2]
#b
asdB <- prop.table(apply(Titanic, c(1,3,4), sum),c(1,2))
asdB[,,2]

#Problem 3
titanic <- data.frame(count = as.vector(Titanic))
dd <- dimnames(Titanic)
titanic$Class <- factor(rep(dd$Class,8))
titanic$Sex <- factor(rep(dd$Sex, each = 4, times = 4))
titanic$Age <- factor(rep(dd$Age, each = 8, times = 2))
titanic$Survived <- factor(rep(dd$Survived, each = 16, times = 1))
#the code transforms the data from a 4D table to a dataframe by segmenting the data by Survived, Age, Sex, Class:
#first is filters the data by Survived Yes/No and then gives us all of the information about it.

#Problem 4
help(mtcars)




