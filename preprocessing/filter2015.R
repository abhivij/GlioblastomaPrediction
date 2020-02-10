library(edgeR)

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
filtered_data <- count_data_2015[keep, ]

y <- DGEList(filtered_data)
y <- calcNormFactors(y)
group <- disease_data_2015$disease_status

design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

par(mfrow=c(1,1))
v <- voom(y,design,plot = TRUE)

normalized_data_2015 <- v$E
colnames(normalized_data_2015) <- disease_data_2015$disease_status

write.csv(normalized_data_2015, "data/output/normalized_GBM_data.csv")

dim(normalized_data_2015)
dim(count_data_2015)
summary(disease_data_2015$disease_status)