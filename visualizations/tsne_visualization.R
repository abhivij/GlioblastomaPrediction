library(Rtsne)

normalized_GBM_data <- read.csv("preprocessing/data/output/normalized_GBM_data.csv", row.names=1)
normalized_GBM_data <- as.data.frame(t(as.matrix(normalized_GBM_data)))

names <- strsplit(rownames(normalized_GBM_data), '\\.')
cancer_flag <- factor(sapply(names, function(x) x[1]))
my_cols <- c('red', 'blue')

iter <- 1
best_tsne <- Rtsne(normalized_GBM_data, dim=2, initial_dims=95, theta=0, max_iter=5000, perplexity=30)
while (iter <= 100) {
  print(iter)
  tsne <- Rtsne(normalized_GBM_data, dim=2, initial_dims=95, theta=0, max_iter=5000, perplexity=30)  
  if (tsne$itercosts[100] < best_tsne$itercosts[100]) {
    best_tsne = tsne
  }
  print(tsne$itercosts[100])
  iter <- iter + 1
}

plot(best_tsne$Y, col=my_cols[cancer_flag])
legend("topleft", c("Cancer", "Non-Cancer"), fill=my_cols, horiz=TRUE)
