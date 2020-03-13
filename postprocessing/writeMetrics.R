acc <- read.csv("../models/model_acc.csv", header=FALSE)
acc_transpose <- as.data.frame(t(as.matrix(acc)))

auc <- read.csv("../models/model_auc.csv", header=FALSE)
auc_transpose <- as.data.frame(t(as.matrix(auc)))


write.csv(acc_transpose, "../models/formatted_acc.csv", row.names=FALSE)

write.csv(auc_transpose, "../models/formatted_auc.csv", row.names=FALSE)
