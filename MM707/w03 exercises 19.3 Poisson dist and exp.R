#Q1
#a
dpois(2, 0.5*8)
#b
ppois(2, 5*0.5)

#Q3
1-ppois(16,12)

#Q4
#case 1 - if emails arrive only during working time
#a
1/(10/12)
#b
3/(10/12)
#c
dpois(6, 10)
#case 2 - if emails arrive all the time, not only working time
#a
#if no emails are rec overnight:
1/(10/24)
#if some emails were received, the time is 0, thus we need to average over both scenario
dpois(0,12*10/24)*2.4*60
#0.97 minutes

#b
#if no email are rec overnight:
3/(10/24)
#if emails were received, the prob that was 1 email
dpois(1,12*10/24)
#the prob that there were 2
dpois(2, 12*10/24)
#the prb that there were more than 3
1-ppois(2, 12*10/24)
#we need to average them:
3/(10/24)*dpois(0,12*10/24)+dpois(1,12*10/24)*(2/(10/24))+dpois(2,12*10/24)*(1/(10/24))
#0.41 hours

#c
#prob does not changes
dpois(6,10)

#Q5
#a
#8 per hour
#b
1-ppois(5,30/6)
#c
#half an hour