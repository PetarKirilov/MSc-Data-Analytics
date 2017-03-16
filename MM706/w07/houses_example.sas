*Example of a SAS program;
ods html close;
ods html;
run;
data houses;
input ref type $ bed heat $ area price;
label bed='no. of bedrooms'
      area='total floor area (square metres)';
datalines;
1  terrace    3  gch   576  155000
2  semi       3  gch   900  230000
3  semi       3  gch   812  275000
4  terrace    3  gch   720  187500
5  semi       4  gf   1040  360000
6  terrace    2  och   550  116000
7  detached   4  gch  1200  420000
8  terrace    3  gch   830  169500
9  endterr    3  ech   610  197500
10 semi       3  ef    850  210000
11 terrace    2  gch   444  149500
12 terrace    3  gf    596  210000
13 semi       3  gch   786  149500
14 semi       3  gch   786  149500
15 endterr    3  ech   650  159500
16 semi       5  gch  1050  325000
17 terrace    3  cch   764  180000
18 detached   3  gch   874  280000
19 detached   4  gch  1450  395000
20 semi       3  gch   920  260000
;
run;

proc print data=houses label; /*”Label” is an option in proc print */
run;

proc means data=houses maxdec=1;
var bed area price;
run;

proc freq data=houses;
table type bed heat;
run;

proc freq data=houses;
table type*bed;
run;

