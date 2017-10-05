data noise;
infile 'C:\Users\Rocky\Desktop\soncheto stochastic MM707\2014T EST.csv' delimiter=',';
input white;
proc print;
run;
proc arima data=noise;
identify var=white;
estimate p=1 noconstant;
forecast lead=10;
run;



data q2;
infile 'D:\University of Brighton\2015-2016 Data Analytics Msc\MM707 Stochastic Methods and forecasting\2014\Q2_2.csv' delimiter=',' firstobs=2;
input obs;
proc print;
run;
proc arima data=q2;
identify var=obs;
estimate p=2 noconstant method=ml;
*forecast lead=10;
run;
