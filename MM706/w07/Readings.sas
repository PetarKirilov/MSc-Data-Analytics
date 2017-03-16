ods html close;
ods html;
run;
data science;
input temp pressure;
datalines;
13  25
14  36
16 39
18  45
22  61
24  65
26 79
;   * End of data ;
run;
proc corr data=sceince plots(only)=scatter;
var temp pressure
run;
