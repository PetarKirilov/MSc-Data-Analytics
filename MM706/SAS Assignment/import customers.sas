* program import and format data;

PROC IMPORT OUT= WORK.Customers 
            DATAFILE= "D:\University of Brighton\2016-2017 Data Analytic
s MSc\2016 MM706 - Programming for Analytics CORE SEM 1 20CR\SAS Assignm
ent\CUSTOMERS.csv" 
            DBMS=CSV REPLACE;
     GETNAMES=YES;
     DATAROW=2; 
RUN;
proc format;
value gend 1 = "male" 2 = "female";
value 
run;
data "D:\University of Brighton\2016-2017 Data Analytics MSc\2016 MM706 - Programming for Analytics CORE SEM 1 20CR\SAS Assignment\customers";
set customers;
run;
data customers;
set "D:\University of Brighton\2016-2017 Data Analytics MSc\2016 MM706 - Programming for Analytics CORE SEM 1 20CR\SAS Assignment\customers";
format gender gend.;
run;
PROC IMPORT OUT= WORK.Orders 
            DATAFILE= "D:\University of Brighton\2016-2017 Data Analytic
s MSc\2016 MM706 - Programming for Analytics CORE SEM 1 20CR\SAS Assignm
ent\ORDERS.csv" 
            DBMS=CSV REPLACE;
     GETNAMES=YES;
     DATAROW=2; 
RUN;
data "D:\University of Brighton\2016-2017 Data Analytics MSc\2016 MM706 - Programming for Analytics CORE SEM 1 20CR\SAS Assignment\orders";
set orders;
run;
PROC IMPORT OUT= WORK.Postcodes 
            DATAFILE= "D:\University of Brighton\2016-2017 Data Analytic
s MSc\2016 MM706 - Programming for Analytics CORE SEM 1 20CR\SAS Assignm
ent\POSTCODES.csv" 
            DBMS=CSV REPLACE;
     GETNAMES=YES;
     DATAROW=2; 
RUN;
data "D:\University of Brighton\2016-2017 Data Analytics MSc\2016 MM706 - Programming for Analytics CORE SEM 1 20CR\SAS Assignment\postcodes";
set postcodes;
run;

* program sort orders;
proc sort data = orders;
by custno;
run;
* program calc_stats;
proc means data = orders noprint;
var actual_order;
by custno;
output out = order_sum n = noorders sum = totalorders mean = meanorders max = maxorders;
run;
proc print data = order_sum; 
run; *to print the order summary;

* program sort data;
proc sort data = postcodes;
by postcode;
run; * sort the postcode by postcode for merging;
proc sort data = customers;
by postcode;
run; *sort the customers data by postcode for merging;
data merged;
merge customers (in=incod) postcodes;
by postcode;
if incod = 1;
run; *merge the postcodes and customers table by postcode;
proc sort data = merged;
by custno;
run; *sort for merging with summary table;
proc sort data = order_sum;
by custno;
run; *sort for merging;
data merged;
merge merged order_sum;
drop _TYPE_ _FREQ_; *excluding the columns that are not needed for the analysis;
run; *the tables are now merged into one;
data "D:\University of Brighton\2016-2017 Data Analytic
s MSc\2016 MM706 - Programming for Analytics CORE SEM 1 20CR\SAS Assignm
ent\merged";
set merged;
run; *save to disk;

*program to analyse results;
*scatter age vs invoice mean;
proc sgplot data = merged;
scatter x = age y = meanorders;
run; 
*correlation between age and mean orders;
proc corr data = merged plots=scatter;
var age meanorders ;
run; 
*regression meanorders vs age;
proc reg data = merged;
model meanorders=age;
run; 
*scatter age vs total value of orders;
proc sgplot data = merged;
scatter x = age y = totalorders;
run;
*scatter age vs number of orders;
proc sgplot data = merged;
scatter x = age y = noorders;
run; 
*scatter age vs max order;
proc sgplot data = merged;
scatter x = age y = maxorders;
run;
*correlation between age and max orders;
proc corr data = merged plots=scatter;
var age maxorders ;
run; 
*regression maxrders vs age;
proc reg data = merged;
model maxorders=age;
run; 
*boxplot of order mean by gender;
proc sgplot data = merged;
vbox meanorders/ category = gender;
run; 
*boxplot of total value of orders by gender;
proc sgplot data = merged;
vbox totalorders/ category = gender;
run;
*barchart of number of orders by gender;
proc sgplot data = merged;
vbar gender/ response = noorders;
run;
*boxplot of max order by gender;
proc sgplot data = merged;
vbox maxorders/ category = gender;
run;
*boxplot of total value of order by region;
proc sgplot data = merged;
vbox totalorders/ category = region;
run;  
*bar chart of number of orders by region;
proc sgplot data = merged;
hbar region/ response = noorders categoryorder = respdesc;
run;
*boxplot;
proc sgplot data = merged;
vbox maxorders/ category = region;
run;
*boxplot of meanorders by region;
proc sgplot data = merged;
vbox meanorders/ category = region;
run; 


