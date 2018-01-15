library(lattice)  # Used for Data Visualization
require(caret)   # for data pre-processing
require(pROC)     # for ROC Curves
library(ipred)    # for bagging and  k fold cv
library(e1071)    # for SVM
print("#################Loading the Credit Card data##################")
data <- read.csv("E:/E drive/SOM/creditcard.csv")
print("#############first 6 observation of the data##########################")
head(data)
attach(data)
library(caret)
set.seed(1234)
splitIndex <- createDataPartition(data$Class, p = .850, list = FALSE, times = 1)
head(splitIndex)
data <- data[ splitIndex,]
testSplit <- data[-splitIndex,]
print(table(data$Class))
print("Let's split the train set into X0(observations belonging to class 0) and X1(observations belonging to class 1")
X0 = data[data$Class == '0',]
X1 = data[data$Class == '1',]
plot(cor(data))
library(cluster)
set.seed(123)
print("Compute and plot wss for k = 2 to k = 15")
k.max <- 50 # Maximal number of clusters
data <- X0
asw <- numeric(50)
for (k in 4:50)
  asw[k] <- clara(data, k) $ silinfo $ avg.width
k.best <- which.max(asw)

plot(4:50, asw[4:50],
     type="l", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")


print("###Clustering using the optimal K(k = 14####")
clustering = clara(data,14)
aggregate(data,by=list(clustering$cluster),FUN=mean)
# append cluster assignment
mydata <- data.frame(data, clustering$cluster)
head(mydata)

cluster1 <- mydata[mydata$clustering.cluster == 1,][,1:31]
cluster2 <- mydata[mydata$clustering.cluster == 2,][,1:31]
cluster3 <- mydata[mydata$clustering.cluster == 3,][,1:31]
cluster4 <- mydata[mydata$clustering.cluster == 4,][,1:31]
cluster5 <- mydata[mydata$clustering.cluster == 5,][,1:31]
cluster6 <- mydata[mydata$clustering.cluster == 6,][,1:31]
cluster7 <- mydata[mydata$clustering.cluster == 7,][,1:31]
cluster8 <- mydata[mydata$clustering.cluster == 8,][,1:31]
cluster9 <- mydata[mydata$clustering.cluster == 9,][,1:31]
cluster10 <- mydata[mydata$clustering.cluster == 10,][,1:31]
cluster11 <- mydata[mydata$clustering.cluster == 11,][,1:31]
cluster12 <- mydata[mydata$clustering.cluster == 12,][,1:31]
cluster13 <- mydata[mydata$clustering.cluster == 13,][,1:31]
cluster14 <- mydata[mydata$clustering.cluster == 14,][,1:31]

train1 = cluster1[sample(nrow(cluster1), 0.0035*nrow(cluster1)), ]
train2 = cluster2[sample(nrow(cluster2), 0.0035*nrow(cluster2)), ]
train3 = cluster3[sample(nrow(cluster3), 0.0035*nrow(cluster3)), ]
train4 = cluster4[sample(nrow(cluster4), 0.0035*nrow(cluster4)), ]
train5 = cluster5[sample(nrow(cluster5), 0.0035*nrow(cluster5)), ]
train6 = cluster6[sample(nrow(cluster6), 0.0035*nrow(cluster6)), ]
train7 = cluster7[sample(nrow(cluster7), 0.0035*nrow(cluster7)), ]
train8 = cluster8[sample(nrow(cluster8), 0.0035*nrow(cluster8)), ]
train9 = cluster9[sample(nrow(cluster9), 0.0035*nrow(cluster9)), ]
train10 = cluster10[sample(nrow(cluster10),0.0035*nrow(cluster10)), ]
train11 = cluster11[sample(nrow(cluster11),0.0035*nrow(cluster11)), ]
train12 = cluster12[sample(nrow(cluster12),0.0035*nrow(cluster12)), ]
train13 = cluster13[sample(nrow(cluster13),0.0035*nrow(cluster13)), ]
train14 = cluster14[sample(nrow(cluster14),0.0035*nrow(cluster14)), ]




print("Merging the samples from the 25 clusters into a single training set")
merged <- Reduce(function(x, y) merge(x, y, all=TRUE), 
                 list(train1, train2, train3, train4, train5, train6, train7, train8, train9, train10, train11, train12, train13,train14))
merged <- merged[complete.cases(merged), ]
head(merged)
print("Let us form the final training data by adding these class 0 samples with the class 1 samples")

trainSplit = merge(merged,X1,all = TRUE)
print(table(trainSplit$Class))

##Lets build c50
### re-train and predict
library(C50)
ctrl <- trainControl(method = "cv", number = 5)
tbmodel <- train(as.factor(Class) ~.-Amount-Time, data = trainSplit, method = "C5.0Tree", trControl = ctrl)
pred <- predict(tbmodel, testSplit)

### score prediction
confusionMatrix(testSplit$Class, pred)


library(pROC)
auc1 <- roc(as.numeric(testSplit$Class), as.numeric(pred))
print(auc1)
plot(auc1, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc1$auc[[1]],3)),col = 'blue')
auc1 <- roc(as.numeric(testSplit$Class), as.numeric(pred),  ci=TRUE)
auc1
plot(auc1, ylim=c(0,1), print.thres=TRUE, main=paste('ROC : Blue is After resampling '),col = 'blue')

library(survival)
library(LogicReg)
library(mcbiopi)
ctrl <- trainControl(method = "cv", number = 5)
modelglm <- train(as.factor(Class) ~.-Amount-Time , data = trainSplit, method = "glm", trControl = ctrl)
summary(modelglm)
### predict
predictors <- names(trainSplit)[names(trainSplit) != 'Class']
predglm <- predict(modelglm, testSplit)
summary(predglm)

confusionMatrix(predglm, testSplit$Class)
aucglm <- roc(as.numeric(testSplit$Class), as.numeric(predglm),  ci=TRUE)
plot(aucglm, ylim=c(0,1), print.thres=TRUE, main=paste('Logistic Regression AUC:',round(aucglm$auc[[1]],3)),col = 'blue')
aucglm
ctrl <- trainControl(method = "cv", number = 5)
modelsvm <- train(as.factor(Class) ~.-Amount-Time, data = trainSplit, method = "svmLinear", trControl = ctrl)
summary(modelsvm)
### predict
predsvm <- predict(modelsvm,testSplit)
summary(predsvm)
### score prediction using AUC
confusionMatrix(predsvm,testSplit$Class)
library(pROC)
aucsvm <- roc(as.numeric(testSplit$Class), as.numeric(predsvm),  ci=TRUE)
plot(aucsvm, ylim=c(0,1), print.thres=TRUE, main=paste('SVM AUC:',round(aucsvm$auc[[1]],3)),col = 'blue')
library(randomForest)
modelrf <- randomForest(as.factor(Class) ~.-Amount-Time , data = trainSplit)
summary(modelrf)
importance(modelrf)
### predict
predrf <- predict(modelrf,testSplit)
summary(predrf)
### score prediction using AUC
confusionMatrix(predrf,testSplit$Class)
library(pROC)
aucrf <- roc(as.numeric(testSplit$Class), as.numeric(predrf),  ci=TRUE)
plot(aucrf, ylim=c(0,1), print.thres=TRUE, main=paste('Random Forest AUC:',round(aucrf$auc[[1]],3)),col = 'blue')
aucrf
library(ggplot2)
plot(aucrf, ylim=c(0,1), main=paste('ROC Comparison : RF(blue),C5.0(black),Adaboost(Green),SVM(Yellow),Logistic(Green)'),col = 'blue')

par(new = TRUE)
plot(auc1)

par(new = TRUE)
plot(aucsvm,col = "yellow")

par(new = TRUE)
plot(aucglm,col = "red")

We find V24 is having the least importance in the Random Forest model, so let's remove them
Random Forest After Feature Engineering
Feature Engineering
importance(modelrf)
modelrf2 <- randomForest(as.factor(Class) ~.-Amount-Time-V24 , data = trainSplit)
summary(modelrf2)


importance(modelrf2)


### predict
predrf2 <- predict(modelrf2,testSplit)
summary(predrf2)

### score prediction using AUC
confusionMatrix(predrf2,testSplit$Class)

library(pROC)
aucrf2 <- roc(as.numeric(testSplit$Class), as.numeric(predrf2),  ci=TRUE)
plot(aucrf2, ylim=c(0,1), print.thres=TRUE, main=paste('Random Forest AUC:',round(aucrf2$auc[[1]],3)),col = 'blue')































