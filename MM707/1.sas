data chemical;
infile 'C:\Users\Rocky\Desktop\soncheto stochastic MM707\\chemical.csv' delimiter=',';
input chem;
time=_n_;
proc print;
run;
proc arima data=chemical;
identify var=chem;
estimate p = 1;
forecast lead=10; 
run;

data bond;
infile 'C:\Users\Rocky\Desktop\soncheto stochastic MM707\BOND (1).csv' delimiter=',';
input bond;
time=_n_;
proc print;
run;
proc arima data=bond;
identify var=bond(1);
estimate p = 1 noconstant;
forecast lead=10; 
run;


data jones;
infile 'C:\Users\Rocky\Desktop\soncheto stochastic MM707\DJONES (1).csv' delimiter=',' firstobs=2;
input date$ dj;
time=_n_;
proc print;
run;
proc arima data=jones;
identify var=dj(1);
estimate q = 1 noconstant method = ml;
forecast lead=10; 
run;


data sasche;
infile 'C:\Users\Rocky\Desktop\soncheto stochastic MM707\sascheto.csv' delimiter=',';
input sasche;
time=_n_;
proc print;
run;
proc arima data=sasche;
identify var=sasche;
estimate q = 1 noconstant;
run;



data house;
infile 'C:\Users\Rocky\Desktop\soncheto stochastic MM707\housing.csv' delimiter=',' firstobs=2;
input date$ house;
time=_n_;
proc print;
run;
proc arima data=house;
identify var=house(1,12);
estimate q=(2)(12) noconstant ;
*estimate q = 1 noconstant;
forecast lead=12  out=table_forecast;
proc print data=table_forecast;
quit;

proc arima data=house;
identify var=house(1,12);
estimate p = (2) q=(12) noconstant ;
*estimate q = 1 noconstant;
forecast lead=12  out=table_forecast;
proc print data=table_forecast;
quit;




data mockche;
infile 'C:\Users\Rocky\Desktop\soncheto stochastic MM707\mockche.csv' delimiter=',';
input mock;
time=_n_;
proc print;
run;
proc arima data=mockche;
identify var=mock;
estimate q = 2;
forecast lead=2;
run;

data dja;
infile 'C:\Users\Rocky\Desktop\soncheto stochastic MM707\DJA_new(1).csv' delimiter=',' firstobs=2;
input dj;
time=_n_;
proc print;
run;
proc arima data=dja;
identify var=dj(1);
estimate q=1 noconstant;
forecast lead=10;
run;
