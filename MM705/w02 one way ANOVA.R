#Q1
amino1 <- spss.get("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 2 - One-way ANOVA/amino.sav")
summary(amino1.model <- aov(amino1$MDA~amino1$AMINO))
#the output from the ANOVA concludes that there is sufficient evidence at the 5% significance level to conclude that the means are different
plot(amino1.model, which = c(1,2,5))
#the assumptions look like they hold. Lets perform the Shapiro Wilk test on the data
shapiro.test(amino1$MDA)
#the shapiro wilk confirms that the data is normal and ANOVA can be performed
pairwise.t.test(amino1$MDA, amino1$AMINO, pool.sd = FALSE, p.adjust.method = "none")
#the pairwise ttest concludes that the means are different between Carnosine and Histidine and between Histidine and Imidazole
kruskal.test(amino1$MDA~amino1$AMINO)
#the test outputs similar results - there is sufficient evidence to believe that the means are different at 5% significance level

#Q2
#0.5%
summary(tablet.model05 <- aov(tablet1$TIME~tablet1$COMPRESS, subset=(tablet1$MS == 0.5)))
#according to ANOVA, there is significant evidence at 5% significance level to believe that the means are different
plot(tablet.model05, which = c(1,2,5))
#the assumptions seem to hold
shapiro.test(tablet1$TIME)
#the shapiro wilk test concludes that the data is not normal but this will not affect much the ANOVA
pairwise.t.test(tablet1$TIME, tablet1$COMPRESS, subset = (tablet1$MS == 0.5), pool.sd = FALSE, p.adjust.method = "none")
#from the pairwise ttest it is evident that there is a bordedline difference between 10 and 20 level of compression, but with the others there is no difference
kruskal.test(tablet1$TIME~tablet1$COMPRESS, subset = (tablet1$MS == 0.5))
#the kruskal wallis confirms that there is a difference in means

#1%
summary(tablet.model1 <- aov(tablet1$TIME~tablet1$COMPRESS, subset = (tablet1$MS == 1)))
#from the ANOVA output, there is significant evidence to believe that the means are different
plot(tablet.model1, which = c(1,2,5))
#plots confirm that the assumptions are confirmed
pairwise.t.test(tablet1$TIME, tablet1$COMPRESS, subset = (tablet1$MS == 1.0), pool.sd = FALSE, p.adjust.method = "none")
#similar results as the previous pairwise ttest
kruskal.test(tablet1$TIME~tablet1$COMPRESS, subset = (tablet1$MS == 1))
#it confirms that there is a difference

#MS is ignored
summary(tablet.modelnoms <- aov(TIME~COMPRESS, data = tablet1))
#ANOVA output concludes that there is unsufficient evidence to reject the null hypothesis of equal means
plot(tablet.modelnoms, which = c(1,2,5))
#assumptions hold
pairwise.t.test(tablet1$TIME, tablet1$COMPRESS, pool.sd = FALSE, p.adjust.method = "none")
#same result
kruskal.test(tablet1$TIME, tablet1$COMPRESS)
#test concluding that the means are different

#Q3
