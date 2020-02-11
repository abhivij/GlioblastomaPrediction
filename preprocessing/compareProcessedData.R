GBM_data <- read.csv("data/output/normalized_GBM_data.csv", stringsAsFactors = FALSE)
NSCLC_data <- read.csv("data/output/normalized_NSCLC_data.csv", stringsAsFactors = FALSE)

GBM_genes <- GBM_data[, 1]
NSCLC_genes <- NSCLC_data[, 1]

common_genes <- Reduce(intersect, list(GBM_genes, NSCLC_genes))

GBM_data <- GBM_data[, -1]
rownames(GBM_data) <- GBM_genes

NSCLC_data <- NSCLC_data[, -1]
rownames(NSCLC_data) <- NSCLC_genes

#filtering out only the common genes
GBM_data <- GBM_data[common_genes, ]

NSCLC_data <- NSCLC_data[common_genes, ]