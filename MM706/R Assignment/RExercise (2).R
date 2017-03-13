dset <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM706 - Programming for Analytics CORE SEM 1 20CR/R Assignment/titanic-b.csv")
#to sort the data by class and name
install.packages("dplyr")
library(dplyr)
dset <- arrange(dset, Class.Dept, Name) #sorting by class and sirname
library(stringr)
splits <- str_split_fixed(dset$Name, ", ", 2) #extracts the name column and convert into surname and the rest of the nanem
splits <- as.data.frame(splits) #converts to dataframe
dset$Surname <- splits$V1 #append the surname column
splits1 <- str_split_fixed(splits$V2, " ", 2) #separate title from first name
splits1 <- as.data.frame(splits1) #convert to dataframe
dset$Title <- splits1$V1 #append the title column
dset$Firstnames <- splits1$V2 #append the first names column
#separate by title
males <- c("Mr")
females <- c("Mrs", "Ms", "Miss")
as.data.frame(t(females))

dset$Gender <- NULL #an example on how to remove a column
dset$Gender <- NA #to add an empty column
dset$Gender[dset$Title == "Mr"] <- "Male" #adds Male to the gender column
dset$Gender[dset$Title == "Ms" | dset$Title == "Mrs" | dset$Title == "Miss"] <- "Female" #adds female to the gender column

nas <- dset[is.na(dset$Gender),] #add the NA's to a new DF
library(gender)
nas$Firstnames <- sub("(\\w+).*", "\\1", nas$Firstnames) #select only first name
nas1 <- gender(as.character(nas$Firstnames), method = 'kantrowitz')
nas2 <- as.data.frame(nas1$name[nas1$gender == "either" | !complete.cases(nas1$gender)]) #extract only NAs and either
nas3 <- gender(as.character(nas2$`nas1$name[nas1$gender == "either" | !complete.cases(nas1$gender)]`), method = "ipums") #leaves out 5 names

