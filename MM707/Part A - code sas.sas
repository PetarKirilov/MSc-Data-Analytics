data prac3;
infile 'D:\MM707\2016-17\W03\Practical - Week 3\prac3.csv' delimiter=',' firstobs=2;
input t s1 s2 s3 s4 s5;
time=_n_;
ods graphics on;
proc print;
run;

%macro arimam(n=);
proc arima data=prac3;
identify var=s&n.;	
run;
%mend;
%arimam(n=1);
%arimam(n=2);
%arimam(n=3);
%arimam(n=4);
%arimam(n=5);
