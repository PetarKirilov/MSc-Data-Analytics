data writing;
infile 'D:\MM365\2015-16\Week5\Data_SAS\writing.csv' dlm=',' firstobs=1;
input paper;
time=_n_;
run;

ods graphics on;

proc arima data=writing
plots=(residual(smooth) forecast(forecasts));
identify var=paper;	
identify var=paper(12);
estimate q=(12);
forecast lead=12  out=outforc;
quit;
proc print;
run;