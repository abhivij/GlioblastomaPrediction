data2017 <- read.table("data/input/2017/GSE89843_TEP_Count_Matrix.txt", header=TRUE)
count_data <- data2017[,-1]
rownames(count_data) <- data2017[,1]

library(openxlsx)
patient_info <- read.xlsx('data/input/2017/mmc2.xlsx', rows=c(3:782), cols=c(2:13), sep.names='_')

healthy_status <- data.frame(factor(patient_info$Classification_group))
rownames(healthy_status) <- gsub('-', '.', patient_info$Sample_name)
colnames(healthy_status) = "status"

library(edgeR)

