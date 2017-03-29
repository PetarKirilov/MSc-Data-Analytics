*Exercise 1;
data hillrace;
infile "C:\Users\ls391\Documents\Brighton\Teaching\2013_2014\MM251\Practicals\Week 5\hillrace.dat";
	*Make sure you use the correct pathname for you;
input location $ 1-25 distance 28-31 height 35-38 time 42-48;
label distance="Distance (miles)"
      time="Time (minutes)";
run;

proc sgplot data=hillrace;
scatter x=distance y=time;
run;

*Not surprisingly, the further they have to run, the longer it takes!;

*Exercise 2;

*Import wine.xls into a SAS data set called wine;

proc sgplot data=wine;
vbox aroma/category=region;

proc sgplot data=wine;
vbox body/category=region;

proc sgplot data=wine;
vbox flavour/category=region;

proc sgplot data=wine;
vbox oakiness/category=region;

proc sgplot data=wine;
vbox oakiness/category=region;
run;

*I'd probably choose the wine from Region 3!;

*Exercise 3;

*Import cities.csv into a SAS data set called cities;

proc print data=cities;
run;

proc sgplot data=cities;
scatter x=workinghours y=salarylevel;
run;

*Longer working hours seem to be associated with lower salries!;

*Exercise 4;

data cambridge;
set "C:\Users\ls391\Documents\Brighton\Teaching\2013_2014\MM251\Practicals\Week 5\cambridge.sas7bdat";
run;

proc print data=cambridge;
run;

proc sgplot data=cambridge;
vbox weight;
run;

*All but one of the crew are heavy - approx 180lbs or above.  One crew member is relatively extremely light;

*You could use File > Export or else use a program such as;

*Exercise 5;
data cambridge;
set "C:\Users\ls391\Documents\Brighton\Teaching\2013_2014\MM251\Practicals\Week 5\cambridge.sas7bdat"; 
file "C:\Users\ls391\Documents\Brighton\Teaching\2013_2014\MM251\Practicals\Week 5\cambridge.dat";
		*Without a delimiter, data will be stored in free-format, ie spaces between values;
put weight;
run;

*Exercise 6a;
Data "C:\Users\ls391\Documents\Brighton\Teaching\2014_2015\MM251\Practicals\Week 5\cities";
set cities;
run;
*Exercise 6b;
Data cities;
set "C:\Users\ls391\Documents\Brighton\Teaching\2014_2015\MM251\Practicals\Week 5\cities";;
run;
proc print;
run;

* Exercise 7;
data xydata;
set "C:\Users\ls391\Documents\Brighton\Teaching\2015_2016\MM706\Week 3\xydata.sas7bdat";
proc means data = xydata;
var x y;
run;
proc sgplot data= xydata;
scatter x=x y=y;
run;
proc reg data=xydata;
model y=x;
run;

/* regression is y = 110 -1.0106x so substitute x = 30 amd the x = 90
