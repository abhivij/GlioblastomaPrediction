# generates GBM and NSCLC normalized data after getting common set of genes

library(edgeR)

data2017 <- read.table("data/input/2017/GSE89843_TEP_Count_Matrix.txt", header=TRUE)
count_data <- data2017[,-1]
rownames(count_data) <- data2017[,1]

disease_data <- data.frame(colnames(count_data))
colnames(disease_data) <- 'sample_name'
disease_data['disease_status'] <- disease_data$sample_name
levels(disease_data$disease_status) <- c(levels(disease_data$disease_status), 'Cancer', 'NonCancer')
disease_data$disease_status[grepl('NSCLC', disease_data$disease_status)] <- 'Cancer'
disease_data$disease_status[grepl('LGG', disease_data$disease_status)] <- 'Cancer'
disease_data$disease_status[!grepl('Cancer', disease_data$disease_status)] <- 'NonCancer'
disease_data$disease_status <- factor(disease_data$disease_status)

keep <- filterByExpr(count_data)
filtered_data <- count_data[keep, ]



count_data_2015 <- read.table("data/input/2015/GSE68086_TEP_data_matrix.txt", header=TRUE)

disease_data_2015 <- data.frame(colnames(count_data_2015))
colnames(disease_data_2015) <- 'sample_name'
disease_data_2015['disease_status'] <- disease_data_2015$sample_name
levels(disease_data_2015$disease_status) <- c(levels(disease_data_2015$disease_status), 'Cancer', 'NonCancer', 'GBM')
disease_data_2015$disease_status[grepl('NSCLC', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status[grepl('CRC', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status[grepl('Chol', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status[grepl('Breast', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status[grepl('BrCa', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status[grepl('Panc', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status[grepl('Lung', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status[grepl('Liver', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status[grepl('GBM', disease_data_2015$disease_status)] <- 'GBM'
disease_data_2015$disease_status[grepl('HD', disease_data_2015$disease_status)] <- 'NonCancer'
disease_data_2015$disease_status[grepl('Control', disease_data_2015$disease_status)] <- 'NonCancer'
disease_data_2015$disease_status[grepl('VU', disease_data_2015$disease_status)] <- 'GBM'
disease_data_2015$disease_status[grepl('6', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status[grepl('1', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status[grepl('5', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status[grepl('3', disease_data_2015$disease_status)] <- 'NonCancer'
disease_data_2015$disease_status[grepl('4', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status <- factor(disease_data_2015$disease_status)

#filtering out non-GBM cancer
other_cancer_filter <- (disease_data_2015$disease_status != 'Cancer')
count_data_2015 <- count_data_2015[, other_cancer_filter]
disease_data_2015 <- disease_data_2015[other_cancer_filter, ]
disease_data_2015$disease_status[grepl('GBM', disease_data_2015$disease_status)] <- 'Cancer'
disease_data_2015$disease_status <- factor(disease_data_2015$disease_status)


keep <- filterByExpr(count_data_2015)
filtered_data_2015 <- count_data_2015[keep, ]

GBM_genes <- rownames((filtered_data_2015))
NSCLC_genes <- rownames(filtered_data)

common_genes <- Reduce(intersect, list(GBM_genes, NSCLC_genes))

#filtering out only the common genes
filtered_data_2015 <- filtered_data_2015[common_genes, ]
filtered_data <- filtered_data[common_genes, ]


# continuing normalization on the common set of genes
y <- DGEList(filtered_data)
y <- calcNormFactors(y)
group <- disease_data$disease_status

design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

par(mfrow=c(1,1))
v <- voom(y,design,plot = TRUE)

normalized_data <- v$E
colnames(normalized_data) <- disease_data$disease_status

write.csv(normalized_data, "data/output/normalized_NSCLC_common_data.csv")

dim(normalized_data)
dim(count_data)
summary(disease_data$disease_status)



y <- DGEList(filtered_data_2015)
y <- calcNormFactors(y)
group <- disease_data_2015$disease_status

design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

par(mfrow=c(1,1))
v <- voom(y,design,plot = TRUE)

normalized_data_2015 <- v$E
colnames(normalized_data_2015) <- disease_data_2015$disease_status

write.csv(normalized_data_2015, "data/output/normalized_GBM_common_data.csv")

dim(normalized_data_2015)
dim(count_data_2015)
summary(disease_data_2015$disease_status)