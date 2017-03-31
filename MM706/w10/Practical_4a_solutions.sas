*Exercise 1;

data steel;
   set "C:\Users\laurie\Documents\Brighton\Teaching\2014_2015\MM251\Practicals\Week 7\steel.sas7bdat";
 
proc print;
run;

*(i);
proc sort data=steel;
by client;

proc means data=steel noprint;
*you can leave out the var statement because you are simply counting number of lines per client;
by client;
output out=njobs n=jobs;

proc print data=njobs;
var client jobs;
run;

*(ii);
proc sort data=steel;
by product;

proc means data=steel noprint;
by product;
output out=nproducts sum=total;

proc print data=nproducts;
var product total;
run;




*Exercise 2;

data flights;
set  "C:\Users\ls391\Documents\Brighton\Teaching\2013_2014\MM251\Practicals\Week 7\flights.sas7bdat";

*(i);
proc sort data=flights;
by date;

proc transpose data=flights out=transposed;
var passengers; 
by date;
id destination;

proc print data=transposed noobs;
run;

*(ii);
proc sort data=flights;
by destination;

proc means data=flights noprint;
output out=stats n=nflights sum=totpass mean=meanpass;
var passengers;
by destination;
run;

proc print data=stats noobs;
var destination nflights totpass meanpass;
format meanpass 7.1; *You can use a format statement in PROC PRINT so that you don't get too many decimal places;
run;

