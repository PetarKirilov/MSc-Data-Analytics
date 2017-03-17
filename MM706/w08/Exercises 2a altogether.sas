* Practical 2 Exercise 1;
proc format;
value genderformat 1="male" 2="female";
value activityformat 1="slight" 2="moderate" 3="a lot";
value levelformat 1="smokes regularly" 2="does not smoke regularly";
data students;
input initial after level gender height weight activity;

diff=after-initial;
bmi=weight/height**2;
label initial='initial pulse rate'
      after='pulse after running'
	  diff='diffrence in pulse rates';
if bmi<=18.5 then bmicat="underweight";
else if bmi<25 then bmicat="normal";
     else if bmi<30 then bmicat="overweight"; 
          else bmicat="obese";

format gender genderformat. level levelformat. activity activityformat.;

datalines;
64	88	2	1	1.68	63.6	2
58	70	2	1	1.83	65.9	2
62	76	1	1	1.87	72.7	3
66	78	1	1	1.85	86.4	1
64	80	2	1	1.75	70.5	2
74	84	2	1	1.85	75.0	1
84	84	2	1	1.83	68.2	3
68	72	2	1	1.88	86.4	2
62	75	2	1	1.83	88.6	2
76	118	2	1	1.80	62.7	2
90	94	1	1	1.88	72.7	1
80	96	2	1	1.83	70.5	2
92	84	1	1	1.78	69.5	3
68	76	2	1	1.70	65.9	2
60	76	2	1	1.80	77.3	3
62	58	2	1	1.83	79.5	3
66	82	1	1	1.75	79.5	2
70	72	1	1	1.85	77.3	3
68	76	1	1	1.88	81.8	2
72	80	2	1	1.68	61.4	3
70	106	2	1	1.80	77.3	2
74	76	2	1	1.78	71.4	2
66	102	2	1	1.78	59.1	2
70	94	1	1	1.90	84.1	2
96	140	2	2	1.55	63.6	2
62	100	2	2	1.68	54.5	2
78	104	1	2	1.73	59.1	2
82	100	2	2	1.73	62.7	2
100	115	1	2	1.60	55.0	2
68	112	2	2	1.78	56.8	2
96	116	2	2	1.73	52.7	2
78	118	2	2	1.75	65.9	2
88	110	1	2	1.75	68.2	2
62	98	1	2	1.59	50.9	2
80	128	2	2	1.73	56.8	2
;

proc means maxdec=2;
var initial after height weight diff bmi;
run;

proc freq data=students;
table level activity bmicat;
run;

proc freq data=students;
table activity*bmicat / nopercent nocol norow;
run;

* Practical 2 Exercise 2;

data prac2ex2;
input date1 anydtdte12. sbp1 dbp1 date2 anydtdte12. sbp2 dbp2;
time_elapsed=datdif(date1,date2,"actual");
format date1 date2 date11.;
datalines;
01-Jun-06	164	97	01-Aug-06	164	93
01-Sep-05	174	121	01-Oct-06	182	108
01-Nov-04	140	80	01-Sep-06	142	84
01-Sep-05	183	104	01-Sep-06	166	106
01-May-06	137	96	01-Oct-06	131	92
01-May-06	140	90	01-Aug-06	136	80
01-Apr-06	141	86	01-Sep-06	110	64
;
run;

proc print data=prac2ex2;
run;

proc means data=prac2ex2 maxdec=1;
var sbp1 dbp1 sbp2 dbp2 time_elapsed;
run;
