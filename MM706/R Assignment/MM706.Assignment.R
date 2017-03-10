#ASSIGNMENT
dat <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM706 - Programming for Analytics CORE SEM 1 20CR/01 R data types and data structures/titanic1.csv")
dat$Name[sample(1:nrow(dat), 10)]
install.packages("gender")
library(gender)
check_genderdata_package()
#nam <- strsplit(as.character(dat$Name), split = ",", fixed = TRUE)

#moves the prefix as first element and Surname as second element
#nam <- sub("(\\w+),\\s(\\w+)","\\2 \\1", dat$Name)
#TODO find a way to separate the nam into different columns with prefix, first, last name

library(stringr)
splits <- str_split_fixed(dat$Name, ", ", 2) #extracts the name column
splits <- as.data.frame(splits) #converts to dataframe
splits1 <- str_split_fixed(splits$V2, " ", 2) #removes the surname
splits1 <- as.data.frame(splits1) #converts to dataframe
firstnames <- splits1$V2 #assigns first names
firstnames <- as.character(firstnames)
library(gender)
gend <- gender(firstnames)     #works but it decreases the names from 2208 to 747
#on the gender function if you select method = 'kantrowitz', it shows all names with NA and the ones which have genders
sex <- gend$gender
sex <- as.data.frame(sex)
summary(sex)

novo <- gender(firstnames, method = 'kantrowitz') #get the genders and NAs
novo1 <- novo[!complete.cases(novo),] #just get the NAs
novo1arr <- arrange(novo1, novo1) #arrange alphabetically

library(tidyr)
a <- separate(dset, Firstname, into = c("Title", "Firstname"), sep = " ")
#to sort
#firstnames <- as.data.frame(firstnames)
#arr <- arrange(firstnames, firstnames)
#ALTERNATIVE
#library(dplyr)
#names <- select(dat, -(Age:Survived))
#library(tidyr)
#names1 <- separate(data = names, col = Name, into = c("lastname", "title"))

#Conveniently, separate() was able to figure out on its own how to separate the sex_class column. Unless you request otherwise with the 'sep' argument, it splits on
#| non-alphanumeric values. In other words, it assumes that the values are separated by something other than a letter or number (in this case, an underscore.)



#USE THIS WHEN DOING COMMENTS ON NEW VARIABLES aql <- melt(airquality) # [a]ir [q]uality [l]ong format