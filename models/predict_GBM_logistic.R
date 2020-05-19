setwd('~/UNSW/VafaeeLab/GlioblastomaPrediction/')

normalized_GBM_data <- read.csv("preprocessing/data/output/normalized_GBM_data.csv", row.names=1)
normalized_GBM_data <- as.data.frame(t(as.matrix(normalized_GBM_data)))

names <- strsplit(rownames(normalized_GBM_data), '\\.')
cancer_flag <- factor(sapply(names, function(x) x[1]))
labels <- as.numeric(cancer_flag) - 1
#normalized_GBM_data$cancer <- as.numeric(cancer_flag) - 1

row_count <- nrow(normalized_GBM_data)
train_index <- sample(1:row_count, 0.8 * row_count)
test_index <- setdiff(1:row_count, train_index)

train_data <- normalized_GBM_data[train_index,]
train_labels <- labels[train_index]
test_data <- normalized_GBM_data[test_index,]
test_labels <- labels[test_index]

train_scaled = scale(train_data)
test_scaled = scale(test_data, center=attr(train_scaled, "scaled:center"), 
                      scale=attr(train_scaled, "scaled:scale"))

train_scaled <- as.data.frame(train_scaled)
train_scaled$cancer <- train_labels
test_scaled <- as.data.frame(test_scaled)
test_scaled$cancer <- test_labels

start <- Sys.time()
model <- glm(cancer ~., family=binomial(link='logit'), data=train_scaled)
print(paste("Time Taken : ", Sys.time() - start))

test_predictions <- predict(model, test_scaled, type='response')
test_prediction_labels <- ifelse(test_predictions > 0.5, 1, 0)

acc <- mean(test_prediction_labels == test_scaled$cancer)
print(paste('Accuracy : ', acc))


library(ROCR)

pr <- prediction(test_predictions, test_scaled$cancer)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc