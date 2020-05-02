normalized_GBM_data <- read.csv("preprocessing/data/output/normalized_GBM_data.csv", row.names=1)
normalized_GBM_data <- as.data.frame(t(as.matrix(normalized_GBM_data)))

names <- strsplit(rownames(normalized_GBM_data), '\\.')
cancer_flag <- factor(sapply(names, function(x) x[1]))
normalized_GBM_data$cancer <- as.numeric(cancer_flag) - 1

row_count <- nrow(normalized_GBM_data)
train_index <- sample(1:row_count, 0.8 * row_count)
test_index <- setdiff(1:row_count, train_index)

train_data <- normalized_GBM_data[train_index,]
test_data <- normalized_GBM_data[test_index,]

start <- Sys.time()
model <- glm(cancer ~., family=binomial(link='logit'), data=train_data)
print(paste("Time Taken : ", Sys.time() - start))

test_predictions <- predict(model, test_data, type='response')
test_prediction_labels <- ifelse(test_predictions > 0.3, 1, 0)

misclass_count <- mean(test_predictions != test_data$cancer)
print(paste('Accuracy : ', 1 - misclass_count))


library(ROCR)

pr <- prediction(test_predictions, test_data$cancer)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc