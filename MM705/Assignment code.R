library(MASS)
library(MVA)
library(Hmisc)
library(psych)
library(GPArotation)
#others
library(Rtsne)
library(cluster)

#task1
dset1 <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Coursework/FINAL/task1samples/sample-57-1.csv")
dset1$X <- NULL
plot(dset1)
pairs(dset1, asp = 1)
PCAdset1 <- princomp(dset1)
summary(PCAdset1)
pairs(PCAdset1$scores, asp = 1)
#this looks like a mixture of three normal distributions
plot3d(dset1)
#dset2
dset2 <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Coursework/FINAL/task1samples/sample-57-2.csv")
dset2$X <- NULL
plot(dset2)
pairs(dset2, asp = 1)
PCAdset2 <- princomp(dset2)
summary(PCAdset2)
pairs(PCAdset2$scores, asp = 1)
plot3d(dset2)
#this looks like a single normal distribution

#task2
emi <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Coursework/FINAL/users.csv")
keys <- as.vector(c("I enjoy actively searching for and discovering music that I have never heard before",
                  "I find it easy to find new music",
                  "I am constantly interested in and looking for more music",
                  "I would like to buy new music but I don't know what to buy",
                  "I used to know where to find music but now I can't find what I want",
                  "I am not willing to pay for music", 
                  "I enjoy music primarily from going out to dance",
                  "Music for me is all about nightlife and going out",
                  "I am out of touch with new music",
                  "My music collection is a source of pride ",
                  "Pop music is fun  it makes me feel good",
                  "Pop music helps me to escape",
                  "I want a multi media experience at my fingertips wherever I go",
                  "I love technology and music is a big part of that technology",
                  "People often ask my advice on music - what to listen to  where to buy it",
                  "I would be willing to pay for the opportunity to buy new music pre-release",
                  "I find seeing a new artist / band on TV a useful way of discovering new music",
                  "I like to be at the cutting edge of new music",
                  "I like to know about music before other people"))
qs <- emi[,c(9:27)]
qs.nona <- na.omit(qs)
for(i in seq_along(qs)){
  Hmisc::label(qs[, i]) <- keys[i]
}
label(qs)

#check the correlation
cortest.bartlett(qs)
KMO(qs)
qs.pca <- principal(qs, nfactors = 19, rotate = "none")
qs.pca$values
plot(qs.pca$values, type = "b")
abline(h=1, col = "red")
#4 is probably the optimal
qs.pca.5 <- principal(qs, nfactors = 5, rotate = "none")
print.psych(qs.pca.5, cut = 0.3, sort = TRUE)
qs.pca.5.varimax <- principal(qs, nfactors = 5, rotate = "varimax")
print.psych(qs.pca.5.varimax, cut = 0.3, sort = TRUE)
fa.diagram(qs.pca.5.varimax, simple = FALSE)
qs.pca.5.oblique <- principal(qs, nfactors = 5, rotate = "oblimin")
print.psych(qs.pca.5.oblique, cut = 0.3, sort = TRUE)
fa.diagram(qs.pca.5.oblique, simple = FALSE)

qs[sample(nrow(qs), 2500),]
plot(qs.ward <- varclus(data.matrix(qs), method = "ward.D2"), hang=-1)
abline(h=1, col="red")
qs.clust <- cutree(qs.ward$hclust, h=1)
#tsnee
plot(qs.tsne <- Rtsne(as.dist(1-rcorr(data.matrix(qs), type = "spearman")$r^2), perplexity = 5)$Y, type = "n")
text(qs.tsne,labels=names(qs.clust),cex=1,col=qs.clust)


plot(varclus(data.matrix(qs), method = "single"), hang=-1)
plot(varclus(data.matrix(qs), method = "complete"), hang=-1)
plot(varclus(data.matrix(qs), method = "ward.D"), hang=-1)
plot(varclus(data.matrix(qs), method = "single"), hang=-1)
qs.ward2 <- varclus(data.matrix(qs),method="ward.D2")
print(qs.clusters.5 <- cutree(qs.ward2$hclust,h=1-(-0.1)))
unname(label(qs)[which(qs.clusters.5==1)])
unname(label(qs)[which(qs.clusters.5==2)])
unname(label(qs)[which(qs.clusters.5==3)])
unname(label(qs)[which(qs.clusters.5==4)])
unname(label(qs)[which(qs.clusters.5==5)])

#task3
library(caret)
set.seed(1604)
brexit <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Coursework/FINAL/BrexitVotingAndWardDemographicData.csv")
str(brexit)
brexit <- brexit[-42,]
brexit$Outcome <- factor(brexit$Outcome)
#brexit$Outcome <- as.numeric(brexit$Outcome)
#brexit$Outcome <- ifelse(brexit$Outcome=="Remain",1,0)
clbrex <- brexit[,c(6, 8, 9, 15:69)]
#clbrex$Outcome <- as.numeric(clbrex$Outcome)
#clbrex[,c(2:58)] <- scale(clbrex[,c(2:58)])
modell <-glm(Outcome~.,data=clbrex,family = binomial(link=logit))

brex2 <- table(clbrex$Outcome, lda(Outcome~., data = clbrex, CV=TRUE)$class)
summary(train(Outcome ~ .,data=clbrex,method="glm",family="binomial",trControl=trainControl(method="cv",number=5)))

bset1 <- brexit[,c(6, 8, 9, 15:21)]
bset2 <- brexit[,c(6, 61:69)]
bset3 <- brexit[,c(6, 56:60)]
#bset1$Outcome <- as.numeric(bset1$Outcome)
#brex1 <- table(bset1$Outcome, lda(Outcome~., data = bset1, CV=TRUE)$class)
#brex1
modell <-glm(Outcome~.,data=clbrex,family = binomial(link=logit))
a <- train(Outcome ~ .,data=bset1,method="glm",family="binomial",trControl=trainControl(method="cv",number=5))
summary(a)
train(Outcome ~ .,data=bset2,method="glm",family="binomial",trControl=trainControl(method="cv",number=5))$results
train(Outcome ~ .,data=bset2,method="glm",family="binomial",trControl=trainControl(method="cv",number=5))$results


a <- principal(clbrex, nfactors=57, rotation="none")
summary(bset1m <- train(Outcome ~ .,data=bset1,method="glm",family="binomial",trControl=trainControl(method="cv",number=5)))
summary(bset1m)

bex <- lda(Outcome~., data=bset1)
bset2$Outcome <- as.numeric(bset2$Outcome)
brex2 <- table(bset2$Outcome, lda(Outcome~., data = bset2, CV=TRUE)$class)
brex2


bset3$Outcome <- as.numeric(bset3$Outcome)
brex3 <- table(bset1$Outcome, lda(Outcome~., data = bset3, CV=TRUE)$class)
brex3





library(Boruta)
brex1.boruta <- Boruta(Outcome~., data=bset1, doTrace = 2)
boruta_signif <- names(brex1.boruta$finalDecision[brex1.boruta$finalDecision %in% c("Confirmed", "Tentative")])
print(boruta_signif)
plot(brex1.boruta, cex.axis=0.5, las=2, xlab="", main="Variable Importance")

brexit$WardCode <- NULL
brexit$WardName <- NULL
brexit$Remain <- NULL
brexit$Leave <- NULL

train(Outcome~., data = brexit, method = "lda", trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5))$results
#wardname <- factor(brexit$WardName)
#countarea <- factor(brexit$CountingArea)
#outcome <- factor(brexit$Outcome)
brexit$WardName <- as.numeric(brexit$WardName)
brexit$WardCode <- as.numeric((brexit$WardCode))
brexit$CountingArea <- as.numeric(brexit$CountingArea)
brexit$Outcome <- as.numeric(brexit$Outcome)
#plot(varclus(data.matrix(brexit)), hang=-1)
br <- brexit[,c(1,3,4,6,7,8,)]
print(brex.lda <- lda(Outcome~.,data=brexit, CV=TRUE))
brex <- table(brexit$Outcome, lda(Outcome~.,data=brexit, CV=TRUE)$class)
brex
#with LDA, the acieved result was pretty good

library(Boruta)
brex.boruta <- Boruta(Outcome~., data=clbrex, doTrace = 2)
boruta_signif <- names(brex.boruta$finalDecision[brex.boruta$finalDecision %in% c("Confirmed", "Tentative")])
print(boruta_signif)
plot(brex.boruta, cex.axis=.6, las=2, xlab="", main="Variable Importance")

clbrex$Outcome <- as.numeric(clbrex$Outcome)
pr <- princomp(clbrex)


library(psych)
clbrex$Outcome <- as.numeric(clbrex$Outcome)
factanal(clbrex, factors=1, rotation="none")
clbrex <- scale(clbrex)




brex3 <- table(clbrex$Outcome, glm(Outcome~No.qualification.Percent+Level.1.qualification.Percent+Level.2.qualification.Percent+Level.3.qualification.Percent+Level.4.or.above.qualification.Percent+Apprenticeship.qualifications.Percent+Other.qualifications.Percent+White.British+White.Irish+White.Gypsy.or.Irish.Traveller+Other.White+Mixed.White.Black.Caribbean+Mixed.White.Black.African+Mixed.White.Asian+Other.Mixed+Indian+Pakistani+Bangladeshi+Chinese+Other.Asian+African+Caribbean+Other.Black+Arabs+Other.Ethnicity, data = clbrex, CV=TRUE)$class)
print(brex3)
summary(train(Outcome ~ No.qualification.Percent+Level.1.qualification.Percent+Level.2.qualification.Percent+Level.3.qualification.Percent+Level.4.or.above.qualification.Percent+Apprenticeship.qualifications.Percent+Other.qualifications.Percent+White.British+White.Irish+White.Gypsy.or.Irish.Traveller+Other.White+Mixed.White.Black.Caribbean+Mixed.White.Black.African+Mixed.White.Asian+Other.Mixed+Indian+Pakistani+Bangladeshi+Chinese+Other.Asian+African+Caribbean+Other.Black+Arabs+Other.Ethnicity,data=clbrex,method="glm",family="binomial",trControl=trainControl(method="cv",number=5)))
train(Outcome ~ No.qualification.Percent+Level.1.qualification.Percent+Level.2.qualification.Percent+Level.3.qualification.Percent+Level.4.or.above.qualification.Percent+Apprenticeship.qualifications.Percent+Other.qualifications.Percent+White.British+White.Irish+White.Gypsy.or.Irish.Traveller+Other.White+Mixed.White.Black.Caribbean+Mixed.White.Black.African+Mixed.White.Asian+Other.Mixed+Indian+Pakistani+Bangladeshi+Chinese+Other.Asian+African+Caribbean+Other.Black+Arabs+Other.Ethnicity,data=clbrex,method="glm",family="binomial",trControl=trainControl(method="cv",number=5))$results
