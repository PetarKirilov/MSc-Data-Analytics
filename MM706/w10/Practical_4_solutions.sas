*Exercise 1;

data cambridge;
set 'C:\Users\ls391\Documents\Brighton\Teaching\2012_2013\MM251\SAS\Week 6\cambridge.sas7bdat';
crew='Cambridge';

data oxford;
set 'C:\Users\ls391\Documents\Brighton\Teaching\2012_2013\MM251\SAS\Week 6\oxford.sas7bdat';
crew='Oxford';

data weights;
set cambridge oxford;

/*Alternative method:
data weights;
set 'H:\MM251\Practicals\Files for practicals\cambridge.sas7bdat'
              'H:\MM251\Practicals\Files for practicals\oxford.sas7bdat';
if _n_ <= 9 then crew='Cambridge';
else crew='Oxford';*/

proc sgplot data=weights;
vbox weight/category=crew;
run;



*Exercise 2;

*Import(File > Import) exhibition.csv and postcodes.csv into SAS data sets exhibition and postcodes;

proc print data=exhibition;
run;

proc print data=postcodes (obs=10); *Note the method of printing just the first 10 observations;
run;

proc sort data=exhibition;
by postcode;

proc sort data=postcodes;
by postcode;

proc format;
value genderfmt 1="Male" 2="Female";
*you could obviously format other variables in a similar way;

data merged;
merge exhibition(in=visitor) postcodes;
by postcode;
format  gender genderfmt. ;
if visitor;
run;

proc print data=merged;
run;



