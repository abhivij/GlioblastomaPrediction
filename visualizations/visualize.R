library(ggplot2)
library(reshape2)

get_lower_tri <- function(cormat){
  cormat[upper.tri(cormat)]<- NA
  return(cormat)
}


normalized_GBM_data <- read.csv("../preprocessing/data/output/normalized_GBM_data.csv", row.names=1)
normalized_GBM_data <- as.data.frame(t(as.matrix(normalized_GBM_data)))

#normalized_GBM_data <- normalized_GBM_data[,1:10]
data_corr <- round(cor(normalized_GBM_data), 2)

#melted_data_corr <- melt(get_lower_tri(data_corr), na.rm=TRUE)

#ggplot(data = melted_data_corr, aes(x=Var1, y=Var2, fill=value)) +
#                  geom_tile() + 
#                  scale_fill_gradient2(name="Pearson\nCorrelation")

pca <- prcomp(normalized_GBM_data, center=TRUE, scale=TRUE)

names <- strsplit(rownames(normalized_GBM_data), '\\.')
cancer_flag <- factor(sapply(names, function(x) x[1]))

ggbiplot(pca, ellipse=TRUE, groups=cancer_flag, var.axes = FALSE)

my_cols <- c('red', 'blue')
pairs(pca$x[, 1:10], pch=19,  cex = 0.5,
      col = my_cols[cancer_flag],
      lower.panel=NULL)