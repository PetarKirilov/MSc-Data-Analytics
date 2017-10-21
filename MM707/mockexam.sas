data dja;
infile 'C:\Users\Rocky\Desktop\soncheto stochastic MM707\DJA_new(1).csv' dlm=',' firstobs=2;
input dja;
run;
ods graphics on;

proc arima data=dja;
identify var=dja;	
run;


proc arima data=dja;
identify var=dja(1);	
run;


proc arima data=dja;
identify var=dja(1);	
estimate q=1 nocostant;
FORECAST LEAD =1 OUT=UTFORC;
proc print;
run;

