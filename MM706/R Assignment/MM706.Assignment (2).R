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
install.packages("stringr")
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
install.packages("tidyr")
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

#NOT USED CODE IN ASSIGNMENT

nas6 <- as.data.frame(gender(as.character(nas5$Firstnames1), method = "ssa"))
nas4 <- within(nas4, rm(Firstnames1,))

dset <- rbind(dset, nas4)

#nas5 <- nas4[nas4$proportion_male > 0.45 & nas4$proportion_male < 0.55,] #extract any particular cases
nas2 <- gender(as.character(nas2$`nas$Firstnames[nas$Gender == "either" | !complete.cases(nas$Gender)]`), method = "ipums")
nas3 <- merge(nas01, nas2, by.x = "Firstnames", by.y = "name")
#23 nas and either

nas3 <- gender(as.character(nas2$`nas1$name[nas1$gender == "either" | !complete.cases(nas1$gender)]`), method = "ipums") #leaves out 5 names
nas4 <- nas1[complete.cases(nas1$gender),]; nas4 <- nas4[nas4$gender != "either",] #assign just male/female
nas4 <- merge(nas01, nas4, by.x = "Firstnames", by.y = "name")
a <- merge(dset, nas, by.x = "Firstnames", by.y = "name")

dset <- rbind(dset, nas)

#nas4 <- nas3[nas3$proportion_female > 0.45 & nas3$proportion_female < 0.55,] #extract any particular cases - Abele is male from the dataset
nass <- nas3[,c(1,4)]
dset$Firstnames <- sub("(\\w+).*", "\\1", dset$Firstnames)
dset$Gender[dset$]

#try to append the found genders 
a <- merge(dset, nass, by.x = "Firstnames", by.y = "name")
a <- a[!complete.cases(a$Gender),]
a$Gender <- nass[match(a$Firstnames, nass$name), 2]

nas33 <- as.vector(nas3$name)
as <- dset[dset$Firstnames %in% nas33 & is.na(dset$Gender),] #match the missing genders
as <- dset[dset$Firstnames %in% nas33 & is.na(dset$Gender),16] #select just gender
dset[dset$Firstnames %in% nas33 & is.na(dset$Gender),16] <
  
  
  
  #to sort the data by class and name
  install.packages("dplyr")
library(dplyr)
#dset <- arrange(dset, Class.Dept, Name) #sorting by class and sirname