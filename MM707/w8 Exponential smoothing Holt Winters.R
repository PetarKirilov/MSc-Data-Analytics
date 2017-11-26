library(fpp)
library(forecast)
qdata <- read.xlsx(file = "D:/Chrome Downloads/qsales for R.xlsx", sheetName = "Sheet2")
#qdata$Years <- as.Date(qdata$Years)
#qdata$Years <- c(2005:2010)
#class(qdata$Years) 
row.names(qdata) <- qdata$Years
qdata <- as.ts(qdata)
qdata <- qdata[,2:5]
fit11 <- hw(qdata, seasonal = "additive")

qhw(austourists, seasonal = "additive")
qdata
aus <- austourists
class(aus)

