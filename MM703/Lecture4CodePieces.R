tt <- c(324,285,708)
lbls <- c("1st","2nd","3rd")
plot(c(0,100),c(0,780),type="n",xlab="Class",ylab="Count",main="Titanic Passengers by Class",xaxt="n",frame.plot=FALSE)
rect(5,0,35,tt[1],col="red")
rect(55,0,65,tt[2],col="green")
rect(75,0,95,tt[3],col="blue")
axis(1, at=c(20,60,85),labels=lbls,lty=0)

barplot(tt,xlab="Class",ylab="Count",main="Titanic Passengers by Class",names.arg=lbls)

par(pty="s",mar=rep(0.1,4))
plot(c(0,100),c(0,100),asp=1,xlab="",ylab="",main="",axes=FALSE,type="n",frame.plot=TRUE)
rect(15,10,35,57,col="black")
rect(65,30,85,80,col="black")

par(pty="s",mar=rep(0.1,4))
plot(c(0,100),c(0,100),asp=1,xlab="",ylab="",main="",axes=FALSE,type="n",frame.plot=TRUE)
rect(15,10,35,70);rect(15,10,35,57,col="black");lines(c(10,40),c(40,40))
rect(65,30,85,90);rect(65,30,85,80,col="black");lines(c(60,90),c(60,60))
abline(h=seq(0,100,by=5),col="black",lty="dotted")

par(pty="s",mar=rep(2,4))
clrs <- c("red","green","blue")
dnst <- c(5,10,20)
pie(tt,labels=paste(lbls," ",round(tt/sum(tt)*100),"%",sep=""),col=clrs,density=dnst,cex=1.3)
barplot(tt,xlab="Class",ylab="Count",names.arg=lbls,col=clrs,density=dnst,cex.names=1.3)

par(pty="s",mar=rep(0.1,4))
sj <- c(39,21.2,3.1,7.4,9.8,19.5)
pie(sj,labels="")
par(pty="s",mar=rep(1,4))
barplot(sj)
abline(h=sj,lty="dotted")

barplot(sj,col=1:length(sj),density=(1:length(sj))+4)
ord <- order(sj,decreasing=TRUE)
barplot(sj[ord],col=ord,density=ord+4)

library(gcookbook)
library(ggplot2)
tophit <- tophitters2001[1:25, ]
nameorder <- tophit$name[order(tophit$lg, tophit$avg)]
tophit$name <- factor(tophit$name, levels=nameorder)
ggplot(tophit, aes(x=name,y=avg,fill=lg)) + 
  geom_bar(stat="identity",width=0.7) + 
  coord_flip() +
  scale_fill_manual(name="League",labels=c("NL","AL"),values=c("red","blue")) +
  ylab("Batting Average") +
  xlab("Hitter") +
  theme_bw() +
  theme(panel.grid.major.y = element_blank(),           # No horizontal grid lines
        legend.position=c(0.5, 0.55),                     # Put legend inside plot area
        legend.justification=c(1, 0.3))
ggplot(tophit, aes(x=avg, y=name)) +
  xlim(0,max(tophit$avg)) +
  geom_segment(aes(yend=name), xend=-0.02, colour="grey50") +
  geom_point(size=3, aes(colour=lg)) +
  scale_colour_manual(name="League",labels=c("NL","AL"),values=c("red","blue")) +
  xlab("Batting Average") +
  ylab("Hitter") +
  theme_bw() +
  theme(panel.grid.major.y = element_blank(),           # No horizontal grid lines
        legend.position=c(0.5, 0.55),                     # Put legend inside plot area
        legend.justification=c(1, 0.3))


png("SunspotsHigh.png",height=250)
plot(sunspot.year,type="l",xlab="",ylab="",main="",frame.plot=FALSE)
lines(lowess(sunspot.year,f=0.15),col="blue",lty="dotted")
dev.off()
png("SunspotsLow.png",height=150)
plot(sunspot.year,type="l",xlab="",ylab="",main="",frame.plot=FALSE)
dev.off()


par(mar=c(2,2,1,1.5))
time <- 2:98
f1 <- log(100/(100-time))
f2 <- log(f1+5)
plot(range(time),c(min(f1),max(f1+f2)),type="n")
polygon(c(time,rev(range(time))),c(f1,0,0),col="blue")
polygon(c(time,rev(time)),c(f1+f2,rev(f1)),col="green")
par(mfrow=c(2,1))
plot(range(time),range(f1),type="n")
polygon(c(time,rev(range(time))),c(f2,0,0),col="green")
plot(range(time),range(f1),type="n")
polygon(c(time,rev(range(time))),c(f1,0,0),col="blue")

par(mar=c(5,4,2,2))
plot(Petal.Length~Petal.Width,data=iris)
plot(Petal.Length~Petal.Width,data=iris,col=c("red","green","black")[iris$Species])
abline(reg=lm(Petal.Length~Petal.Width,data=iris),col="blue",lty="dotted")

distn <- c(2,24,46,239000,93000000)
barplot(log2(distn),horiz=TRUE,names.arg=c("Hove","Seaford","London","Moon","Sun"),xlab="Binary Log(Distance)",las=1)

prbb <- c(0.001,0.01,0.99,0.999)
barplot(prbb,names.arg=prbb,main="As Is")
barplot(prbb,log="y",names.arg=prbb,main="Log")
barplot(log(prbb/(1-prbb)),names.arg=prbb,main="Logit")

par(mfrow=c(1,2))
plot(height~age,data=Loblolly)
abline(reg=lm(height~age,data=Loblolly),col="blue",lty="dotted")
plot(I(height^2)~age,data=Loblolly,ylab="height^2")
abline(reg=lm(I(height^2)~age,data=Loblolly),col="blue",lty="dotted")

par(mfrow=c(1,1))
plot(Carbon~City,data=fpp::fuel,xlab="Miles per Gallon",ylab="CO2 footprint")
lines(lowess(fpp::fuel$Carbon~fpp::fuel$City),col="blue",lty="dotted")
plot(log(Carbon)~log(City),data=fpp::fuel,xlab="Log(Miles per Gallon)",ylab="log(CO2 footprint)")
lines(lowess(log(fpp::fuel$Carbon)~log(fpp::fuel$City)),col="blue",lty="dotted")

library(PerformanceAnalytics)
chart.Correlation(iris[-5], bg=iris$Species, pch=21)
pairs(iris[-5], col=iris$Species)
