planets <- read.csv("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM706 - Programming for Analytics CORE SEM 1 20CR/04 R drawing and plotting/planets.csv")
plot(0,0) #datapoint with coordinates 0,0
plot(0,0, xlim = c(-10, 10), ylim = (c(-10, 10))) #add limits to the plot so that we can visualise what we want 
                                                  #(if we want to draw a seqence of plots; or we want our limits to be symmetrical)
plot(0,0, xlim = c(-10, 10), ylim = c(-10, 10), xlab = "x", ylab = "y") #add the labels
points(planets$x, planets$y, pch = 16)
col <- c("yellow","black","orange","blue","red","orange","gray") #use colours to colour up our data for easy representation
pch <- c(8,16,16,10,16,16,16)
plot(x=NULL ,xlim=c(-10,10),ylim=c(-10,10),xlab="x / AU",ylab="y / AU")
points(planets$x,planets$y,pch=pch ,col=col)
#lets think about sizes now
# setting planets ' colors
col <- c("yellow","black","orange","blue","red","orange","gray")
# setting planets ' symbols
pch <- c(8,16,16,10,16,16,16)
# setting planets ' sizes (TODO: automate using radius)
#may use the logarith scale to map them to sizeble sizes - compress it as we have a big variety of sizes; to convey the idea of relative sizes but sacrifice the precision
size <- c(2,0.5,1,1,0.5,2,2)
# create empty plot
plot(x=NULL ,xlim=c(-10,10),ylim=c(-10,10),xlab="x / AU",ylab="y / AU")
 # put points on plot
points(planets$x,planets$y,pch=pch ,col=col ,cex=size)
# put points for the legend
points(x=rep(7,7),y=10:4,pch=pch ,col=col ,cex=size)
# put legend labels
text(x=rep(7.5,7),y=10:4, planets$name ,adj=0)
# create legend boundary
rect(5,3,10.5,10.5)




#problem 1 c
plot(x = NULL, y = NULL, xlim = c(-1,1), ylim = c(-1,1), xlab = "x", ylab = "y")
c <- seq((1/6)*pi, 2*pi, (1/6*pi))
na <- c(1:12)
points(0.9*sin(c), 0.9*cos(c))
text(sin(c), cos(c), na)
arrows(0,0,0.8*sin(c[12]),0.8*cos(c[12]))
arrows(0,0,0.5*sin(c[10]),0.5*cos(c[10]))

#d 
h <- format(Sys.time(), "%H")
m <- format(Sys.time(), "%M")
h <- as.integer(h)
m <- as.integer(m)
plot(x = NULL, y = NULL, xlim = c(-1,1), ylim = c(-1,1), xlab = "x", ylab = "y")
c <- seq((1/6)*pi, 2*pi, (1/6*pi))
na <- c(1:12)
points(0.9*sin(c), 0.9*cos(c))
text(sin(c), cos(c), na)
arrows(0,0,0.8*sin(m*2*pi/60),0.8*cos(m*2*pi/60))
arrows(0,0,0.5*sin(c[h]),0.5*cos(c[h]))

#problem 2
# Set number of nodes
n <- 5
# Get the coordinates of the nodes at random (uniformly on [0,1])
x <- runif(n)
y <- runif(n)
# Generate the population at random (integer up to 1000)
p <- as.integer(1000*runif(n))
# Generate the connection graph: 0 means no link , 1 means there is a link
# We need n*(n-1)/2 numbers by the number of possible links
# sample picks 0 with probability 0.3 and 1 with probability 0.7
d <- sample(c(0,1),prob=c(0.3,0.7), n*(n-1)/2,replace=TRUE)
# Generate traffic jams: integers from 1 to 10
d <- d*as.integer(1+10*runif(n*(n-1)/2))
# Convert the traffic jam vector into symmetrical matrix
m <- matrix(0, nrow = n, ncol = n)
m[upper.tri(m)] <- d
m <- m + t(m)
m
# Convert population to point size (log-scale)
cex <- (1+log10(p))/2
# Convert traffic jam parameter matrix into the dataframe
t <- m/10 # rescale to [0,1]
t <- data.frame(jam=as.vector(t)) # stretch into the vector
t$i <- rep(1:n,times=n,each=1) # i-node
t$j <- rep(1:n,times=1,each=n) # j-node
# Keep only links with i>j (the half of the matrix)
t <- t[t$i>t$j,]
# Remove links with no roads
t <- t[t$jam>0,]
# Convert jam to color (black: calm , red: jammed)
t$col <- rgb(t$jam,0,0)
# Create empty plot
plot(x=NULL,xlim=c(0,1),ylim=c(0,1),xlab="x",ylab="y")
# Connect nodes with roads
segments(x0=x[t$i],y0=y[t$i],x1=x[t$j],y1=y[t$j],col=t$col)
# Draw nodes
points(x,y,cex=cex,pch=16)
# Draw node numbers
text(x,y+0.05,labels=1:n,col="blue")


#problem3
hist(iris$Petal.Length)
irises <- split(iris, iris$Species)
setosa <- iris[iris$Species == 'setosa',]
versicolor <- iris[iris$Species == 'versicolor',]
virginica <- iris[iris$Species == 'virginica',]
col <- c( rgb(1,0,0, 0.5), rgb(0,1,0, 0.5), rgb(0,0,1, 0.5))
hist(setosa$Petal.Length, add = TRUE, col = col[1])
hist(versicolor$Petal.Length, add =TRUE, col = col[2])
hist(virginica$Petal.Length, add=TRUE, col = col[3])
legend(4,35,pch=15,cex=1.5,col=col,legend=levels(iris$Species)) 