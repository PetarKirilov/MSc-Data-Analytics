*Exercise 1;
data double;
rate=0.05;
* use a variable for the rate so we can solve for other interest rates;
invested=100;
amount=invested;
count=0;
do while (amount<invested*2);
  count=count+1;
  amount=amount*(1+rate);
end;
output;
proc print;
run;

*Exercise 2
ods html close;
ods html;
run;
data loan;
balance=5000;
repayment=200;
interest=0.005;
count=0;
do while (balance>0);
  count=count+1;
  balance=balance*(1+interest);
  if balance<200 then repayment=balance; 
  balance=balance-repayment;
  output;
end;
output;
proc print data=loan;
run;


*Exercise 3;
data home;
sum=0;
count=0;
* loop until 20 as an example,could be any number;
do while (count<20);
  count=count+1;
  next_term=1/(2**count);
  sum=sum+next_term;
end;
output;
proc print;
run;


*Exercise 5;
ods html close;
ods html;
run;
data divide;
a=20;
b=3;
do while (a>=b);
  count=count+1;
  a=a-b;
  output;
end;
remainder=a;
output;
proc print data=divide;;
run;

