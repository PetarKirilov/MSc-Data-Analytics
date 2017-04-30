n <- 20
x <- runif(n)
y <- runif(n)
plot(x,y)
n <- 20
x <- runif(n)
y <- x^2
plot(x,y)


x <- seq(from = 0, to = 10, by = 0.01)
y <- x
lines(plot(x, sin(y)))
points(plot(x, sin(y)))


#ex2
t <- seq(0, 10, length = 128)
f <- dunif(t, min = 2, max = 6)
plot(f)
x <- runif(10^3, min = 2, max = 6)
hist(x)
plot(density(x))
points(density(x))
plot(density(f))


#ex3
x <- runif(1000, 0, 1)
y <- runif(1000, 0, 1)
z <- pmin(x, y)
plot(density(z))
mean(z)


#ex4
x <- runif(1000, 0, 1)
y <- runif(1000, 0, 1)
z <- runif(1000, 0, 1)
new <- x+y+z
plot(density(new))
t <- seq(0, 3, length = 100)
f <- dnorm(t, mean(t), sd(t))
plot(density(f))
plot(density(new))
