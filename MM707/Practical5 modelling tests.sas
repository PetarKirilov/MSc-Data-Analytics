ods graphics on;
data housing;
infile 'D:\MM365\2016-17\Week5\Lab\housing.csv' dlm=',' firstobs=2;
input date $ sales;
run;

proc arima data=housing
plots=(residual(smooth) forecast(forecasts));	
identify var=sales(1,12);
estimate q=(1,2)(12) noconstant ; /*2 MA non-seas + 1 MA seas*/
estimate q=(2)(12) noconstant ; /*using a subset for MA non-seas*/
forecast lead=12  out=table_forecast;
proc print data=table_forecast;
quit;

/*We have also check for:*/
proc arima data=housing
plots=(residual(smooth) forecast(forecasts));	
identify var=sales(1,12);
estimate p=(1,2)(12) noconstant ;
estimate p=(2) q=(12) noconstant ;
forecast lead=12  out=table_forecast;
quit;
proc print data=table_forecast;
	run;
