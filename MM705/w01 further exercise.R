#exercise 1
library(Hmisc)
library(BSDA)
pulse <- spss.get("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 1 - One and Two sample tests/pulse.sav")
shapiro.test(pulse$pulse1)
#according to the Shapiro Wilk, the first pulse data is not normal as the p-value is less than the critical 5% value
shapiro.test(pulse$pulse2)
#same for the second puls - it confirmed the expected non-normality
#Testing for pulse1 mean = 75
wilcox.test(pulse$pulse1, mu = 75)
#the above Wilcoxon test, the null hypothesis can be accepted - there is insufficient evidence at the 5% level to conclude that there is a difference in means
SIGN.test(pulse$pulse1, md = 75)
#However, from the Sign test it can be concluded that the null hypothesis can be rejected

#Testing for pulse1 > 72
wilcox.test(pulse$pulse1, mu = 72, alternative = "g")
#the Wilcoxon test accepts the null hypothesis at 5% level, concluding that the resting pulse is above not 72
SIGN.test(pulse$pulse1, md = 72, alternative = "g")
#similarly the Sign test concludes that the mean is not greater than 72

#Testing for pulse1 < 80
wilcox.test(pulse$pulse1, mu = 80, alternative = "l")
#the output from the Wilcoxon test concludes that the mean is below 80, thus rejecting the null hypothesis at the 5% level
SIGN.test(pulse$pulse1, md = 80, alternative = "l")
#similar output with the Sign test.

#Testing for pulse2 mean = 75
wilcox.test(pulse$pulse2, mu = 75)
#the above Wilcoxon test, the null hypothesis can be rejected - there is sufficient evidence at the 5% level to conclude that there is a difference in means
SIGN.test(pulse$pulse2, md = 75)
#However, from the Sign test it can be concluded that the null hypothesis can be accepted

#Testing for pulse2 > 72
wilcox.test(pulse$pulse2, mu = 72, alternative = "g")
#the Wilcoxon test rejects the null hypothesis at 5% level, concluding that the resting pulse is above 72
SIGN.test(pulse$pulse2, md = 72, alternative = "g")
#similarly the Sign test concludes that the mean is greater than 72

#Testing for pulse2 < 80
wilcox.test(pulse$pulse2, mu = 80, alternative = "l")
#the output from the Wilcoxon test concludes that the mean is not below 80, thus accepting the null hypothesis at the 5% level
SIGN.test(pulse$pulse2, md = 80, alternative = "l")
#similar output with the Sign test.

#Testing for Females
shapiro.test(pulse$pulse1[sex ==  "female"])
#the Shapiro Wilk test concludes that the data is normal, thus we will use t.test
t.test(pulse$pulse1[sex == "female"], mu = 75)
#the t-test concludes that there is sufficient evidence to reject the null hypothesis, thus the mean is not equal to 75
t.test(pulse$pulse1[sex == "female"], mu = 72, alternative = "g")
#the results from the t-test conclude that the mean is no greater than 72
t.test(pulse$pulse1[sex == "female"], mu = 80, alternative = "l")
#the results from the t-test conclude that the mean is less than 80 as there is sufficien evidende at 5% level of significance to conclude

#Testing for Males
shapiro.test(pulse$pulse1[sex == "male"])
#the Shapiro Wilk test concludes that the data is not normal, thus we will use non-parametric tests
wilcox.test(pulse$pulse1[sex == "male"], mu = 75)
#the wilcoxon test concludes that there is insufficient evidence at 5% level of significance to conclude that the mean is not equal to 75
SIGN.test(pulse$pulse1[sex == "male"] , md = 75)
#the above conclusion is also confirmed by the Sign test
wilcox.test(pulse$pulse1[sex == "male"], mu = 72, alternative = "g")
#the Wilcoxon test concludes that there is insufficient evidence at 5% significance level to conclude that the mean is greater than 72
SIGN.test(pulse$pulse1[sex == "male"], md = 72, alternative = "g")
#the Sign test confirms the wilcoxon test
wilcox.test(pulse$pulse1[sex == "male"], mu = 80, alternative = "l")
#the wilcoxon test gives sufficient evidence at the 5% significance level to conlcude that the mean is less than 80
SIGN.test(pulse$pulse1[sex == "male"], md = 80, alternative = "l")
#the sign test confirms the above conclusion

#Testing for comparison between ran vs not ran
shapiro.test((pulse$pulse2 - pulse$pulse1)[unclass(pulse$ran)==1])
#the Shapiro Wilk test concludes that the data is normal, thus we will use t.test
t.test((pulse$pulse2-pulse$pulse1)[unclass(pulse$ran)==1], alternative = "g")
#the t test concludes that there is difference between means of people who ran and did not run
qqnorm((pulse$pulse2-pulse$pulse1)[unclass(pulse$ran)==1])
qqline(c((pulse$pulse2-pulse$pulse1)[unclass(pulse$ran)==1]))

#Testing for comparison between smokers and non-smokers
shapiro.test(pulse$pulse1[unclass(pulse$smokes) == 1])
shapiro.test(pulse$pulse1[unclass(pulse$smokes) == 2])
#both datasets conform to the normality assumption, thus we will use t.tests
t.test(pulse$pulse1[unclass(pulse$smokes) == 1],pulse$pulse1[unclass(pulse$smokes) == 2])
#the t test concludes that the means are no different

var.test(pulse1~sex, data = pulse)
#the variance homogeneity concludes that the variances are equal
t.test(pulse$pulse1~pulse$sex,pulse$pulse, var.equal=T)
#the above estimates the variances


#Q2
tablet1 <- spss.get("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 1 - One and Two sample tests/tablets.sav")
tablet1$MS <- as.factor(tablet1$MS)
tablet1$COMPRESS <- as.factor(tablet1$COMPRESS)
shapiro.test(tablet1$TIME)
#from the Shapiro Wilk test it is evident that the data is no normal and we need to use non-parametric tests
wilcox.test(tablet1$TIME[tablet1$MS == 0.5], tablet1$TIME[tablet1$MS == 1])
#there is a difference in the disintegration times
SIGN.test(tablet1$TIME[tablet1$MS == 0.5], tablet1$TIME[tablet1$MS == 1])
#sign test confirms it
wilcox.test(tablet1$TIME[tablet1$COMPRESS == 20]~tablet1$MS[tablet1$COMPRESS == 20], alternative = "g")
#it seems that disintegration times do not increase, thus accepting the null hypothesis
SIGN.test(tablet1$TIME[tablet1$COMPRESS == 20 & tablet1$MS == 0.5], tablet1$TIME[tablet1$COMPRESS == 10 & tablet1$MS == 1])
wilcox.test(tablet1$TIME[tablet1$COMPRESS == 20 & tablet1$MS == 0.5],tablet1$TIME[tablet1$COMPRESS == 10 & tablet1$MS == 1])
#Q3
basket <- spss.get("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Week 1 - One and Two sample tests/basketball.sav")
shapiro.test(basket$BEFORE)
#the data is normal
qqnorm(basket$BEFORE)
qqline(c(basket$BEFORE))
shapiro.test(basket$AFTER)
qqnorm(basket$AFTER)
qqline(basket$AFTER)
#the after data does not look really normal
t.test(basket$BEFORE, basket$AFTER, paired = TRUE)
wilcox.test(basket$BEFORE, basket$AFTER, paired = TRUE)
SIGN.test(basket$BEFORE, basket$AFTER)
boxplot(basket)
#from the above output, it can be concluded that there is sufficient evidence at the 5% significance level to conclude that there is a difference in means
kruskal.test(basket$BEFORE, basket$AFTER)
#the kruskal wallis non-parametric test concludes that there is no difference