install.packages("ggplot2")
require(ggplot2)
str(diamonds)
ggplot(data = diamonds) + aes(x = carat, y = price, colour = clarity) + theme_bw() + scale_x_log10() + scale_y_log10() + geom_point(size = 0.1) + guides(colour = guide_legend(override.aes = list(size = 1)))