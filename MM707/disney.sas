data disney;
infile 'D:\MM707\2016-17\W05\disney.csv' dlm=',' firstobs=2;
input sales date yyQ4.;
logsales=log(sales);
format date yyQ4.;
proc print;
run;
ods graphics on;
proc sgplot;
scatter x=date y=sales;
run;

/* try generating date? eg

      date = intnx( 'month', '31dec1948'd, _n_ );
      format date monyy.;
*/


proc arima data=disney
plots(unpack);
identify var=sales;	
run;

proc arima data=disney
plots(unpack);
identify var=logsales;	
run;

proc arima data=disney;
identify var=logsales;	
run;

proc arima data=disney;
identify var=logsales(4);	
run;

identify var=logsales(4);
estimate q=1;
run;

proc arima data=disney
plots=(residual(smooth) forecast(forecasts));
identify var=logsales(4);
estimate q=(1)(4);
forecast lead=4  out=outforc;
proc print;
run;
