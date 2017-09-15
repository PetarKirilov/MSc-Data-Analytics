library(psych)
judge <- USJudgeRatings
head(judge)
pairs(judge, asp = 1)
factanal(judge, factors = 4, rotation = "none")
#contact with layers cannot be explained at all; first factor is the combination of all qualities; second factor is again a combination, but a smaller one of some of their qualities
factanal(judge, factors = 5, rotation = "none")
#fifth factor is the number of contacts to lawyers
factanal(judge, factors = 4, rotation = "varimax")
#similar results as the previous factor analysis

factanal(judge, factors = 2, rotation = "none")
fa(judge,nfactors=2,rotate="none")
judgecomp <- principal(judge, nfactors = 2, rotate="none")
plot(judgecomp$values,type="b")
#2 factors are the optimal variant, with the first one explaining 95% of the variability
fa.diagram(judgecomp, simple = FALSE)

bf <- bfi
head(bfi)
bf.princomp <- principal(bf, nfactors = 8, rotate = "none")
plot(bf.princomp$values, type="b")
#looks like 8 factors or less are most optimal
fa(bf, nfactors = 8, rotate = "none")
#8 factors look like too much judging my the proportion explained; we can try with 6 factors
fa(bf, nfactors = 6, rotate = "none")
fa.parallel(bf)
#this analysis reports that 7 or less is optimal
fa(bf, nfactors = 4, rotate = "none")
