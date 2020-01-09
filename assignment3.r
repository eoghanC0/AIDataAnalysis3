library(matrixStats)
library(plyr)
library("ggplot2")
install.packages("UsingR")
library(UsingR)
install.packages("visualize")
library(visualize)
install.packages("class")
library(class)
install.packages("caret")
library(caret)


#create initial data frames of the three group sets
df <- read.table("/home/eoghan/CSC3060/Assignment 2/section2_features/40181561_features.tsv", header = TRUE)
head(df)
living <- c("banana", "cherry", "flower", "pear")
nonliving <- c("envelope", "golfclub", "pencil", "wineglass")
living_df <- df[df$label %in% living,]
nonliving_df <- df[df$label %in% nonliving,]

#put all new training items into a dataset

temp = list.files(path="/home/eoghan/CSC3060/Assignment 3/UlFl0FinFmKZlHd/", pattern="*features.csv")

dataset <- data.frame(matrix(0, ncol = 22, nrow = 0))
colnames(dataset)<-c("label", "index", "nr_pix", "height", "width", "span", "X5rows", "X5cols", "neigh1", "neigh5", "left2", "right2", "top2", "bott2", "vert", "horiz", "rDiag", "lDiag", "regions", "eyes", "hollow", "tile2")

temp_dataset <- data.frame(matrix(0, ncol = 22, nrow = 0))
colnames(temp_dataset)<-c("label", "index", "nr_pix", "height", "width", "span", "X5rows", "X5cols", "neigh1", "neigh5", "left2", "right2", "top2", "bott2", "vert", "horiz", "rDiag", "lDiag", "regions", "eyes", "hollow", "tile2")

getwd()
setwd("/home/eoghan/CSC3060/Assignment 3/UlFl0FinFmKZlHd/")

for (file in temp) {
  temp_dataset <-read.table(file, header=FALSE, sep="\t")
  colnames(temp_dataset)<-c("label", "index", "nr_pix", "height", "width", "span", "X5rows", "X5cols", "neigh1", "neigh5", "left2", "right2", "top2", "bott2", "vert", "horiz", "rDiag", "lDiag", "regions", "eyes", "hollow", "tile2")
  dataset<-rbind(dataset, temp_dataset)
  #rm(temp_dataset)
}

############# #Q1 #############

#create living column which will be 1 if item is a living item and 0 if non-living
df$living <- 0
df$living[df$label %in% living] <- 1

#create living column which will be 1 if item is a living item and 0 if non-living
dataset$living <- 0
dataset$living[dataset$label %in% living] <- 1

require(ISLR)
install.packages("corrplot")
library(corrplot)

correlations <- cor(df[,3:22])
corrplot(correlations, method="circle")

new_dataset <- df[,c("vert", "living")]
new_dataset

#logistic regression
glm.fit <- glm(living ~ vert, data = new_dataset, family = binomial)
summary(glm.fit)

plt <-ggplot(new_dataset, aes(x=vert, y=living)) + 
  geom_point(aes(colour = factor(living)), 
             show.legend = T, position="dodge")+
  geom_line(data=fitted.curve, colour="orange", size=1)

plt

glm.fit$coefficients

beta0 = as.numeric(glm.fit$coefficients[1])
beta1 = as.numeric(glm.fit$coefficients[2])
beta0
beta1

# recall the logistic regression formula from lectures

linear_combo = beta0 + beta1*1.555

estimated_probability = exp(linear_combo)/(1+exp(linear_combo))

estimated_probability

# Prediction - using the `predict` function 
# =========================================
# To plot the fitted curve, let's make a data frame containing the predicted 
# values across the range of feature values (i.e. across the x-axis)

x.range = range(new_dataset[["vert"]])
x.range

# the `seq` command allows values across a range:
?seq

x.values = seq(x.range[1],x.range[2],length.out=1000)
x.values
library(matrixStats)

########### Q2 #############

# Assuming a p>0.5 cut-off, calculate accuracy on the training data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tail(new_dataset)
new_dataset[["predicted_val"]] = predict(glm.fit, new_dataset, type="response")
new_dataset[["predicted_class"]] = 0
new_dataset[["predicted_class"]][new_dataset[["predicted_val"]] > 0.5] = 1

correct_items = new_dataset[["predicted_class"]] == new_dataset[["living"]] 
correct_items
# proportion correct:
nrow(new_dataset[correct_items,])/nrow(new_dataset)

# proportion incorrect:
nrow(new_dataset[!correct_items,])/nrow(new_dataset)


########### Q3 #############

new_dataset <- df[,c("label", "living", "nr_pix", "height", "width")]
new_dataset

glm.fit <- glm(living ~ nr_pix + height + width, data = new_dataset, family = binomial)
summary(glm.fit)

new_dataset[["predicted_val"]] = predict(glm.fit, new_dataset, type="response")
new_dataset[["predicted_class"]] = 0
new_dataset[["predicted_class"]][new_dataset[["predicted_val"]] > 0.5] = 1

correct_items = new_dataset[["predicted_class"]] == new_dataset[["living"]] 
correct_items
# proportion correct:
nrow(new_dataset[correct_items,])/nrow(new_dataset)

# proportion incorrect:
nrow(new_dataset[!correct_items,])/nrow(new_dataset)

library(caret)

train_control <- trainControl(method="cv", number=5)
model <- train(as.factor(living)~nr_pix + height + width, data=new_dataset, trControl=train_control, method="glm", family=binomial)
print(model)

############## Q4 ##############
set.seed(3060)
randomModelData <- new_dataset
randomModelData
randomModelData$predicted_class <- sample(c(0,1), replace = TRUE, size = 160)
correct_items = randomModelData[["predicted_class"]] == randomModelData[["living"]] 
correct_items
# proportion correct:
nrow(randomModelData[correct_items,])/nrow((randomModelData))

# proportion incorrect:
nrow(randomModelData[!correct_items,])/nrow(randomModelData)

#p-value for getting 80 items correct with a probabiity of 71.25 
1 - pbinom(47.5, 100, 0.5)


############# Q5 ##############
confusionMatrix(model)

#######################################        PART 2       ##############################################
#part 2 q 1 : use first 8 features to predict label
dataset
train.X = cbind(dataset$nr_pix, dataset$height, dataset$width, dataset$span, dataset$X5rows, dataset$X5cols, dataset$neigh1, dataset$neigh5)
colnames(train.X)<-c("nr_pix", "height", "width", "span", "X5rows", "X5cols", "neigh1", "neigh5")
summary(train.X)
head(train.X)

train.X <- as.data.frame(train.X)

train.Labels = dataset$label

accuracyDF = data.frame(matrix(ncol = 2, nrow = 0))

for (i in seq(1, 59, 2)) {
  knn.pred=knn(train.X,train.X,train.Labels,k=i)
  print(table(knn.pred,dataset$label))
  correct_list = knn.pred == train.Labels
  nr_correct = nrow(train.X[correct_list,])
  
  acc_rate = nr_correct/nrow(train.X)
  accuracyDF <- rbind(accuracyDF, cbind(i, acc_rate))
}

ggplot(accuracyDF, aes(x=i, y=acc_rate)) + geom_line(color = 'red') + labs(x = 'neighbours', title = 'Model accuracy rate with KNN increasing number of neighbours')

################# Q2 #################

train.X <- cbind(train.X, dataset$label)
names(train.X)[9]<-paste("label")
train_control <- trainControl(method = "cv", number = 5)
model <- train(label ~ ., data  = train.X, method = "knn", trControl = train_control, tuneGrid = expand.grid(k = seq(1, 59, 2)))
model
plot(model)
plotDf <- data.frame(matrix(ncol = 3, nrow = 0))
colnames(plotDf) <- c("k", "accuracy1", "accuracy2")
plotDf

accuracyDF <- cbind(accuracyDF, model$results[2])
accuracyDF

p <- ggplot(accuracyDF, aes(x=1/i)) + geom_line(aes(y = 1-acc_rate, color = 'red')) + geom_point(aes(y = 1-acc_rate, color = 'red')) + geom_line(aes(y = 1 -Accuracy, color = 'blue')) + geom_point(aes(y=1-Accuracy, color = 'blue')) + labs(x = '1/k', y = 'Error Rate', title = 'Error rates for classification and cv classification models')
p  + scale_colour_discrete(name = "legend", labels = c("cv", "training set"))

############### Q3 ################
confusionMatrix(model)

#######################################        PART 3       ##############################################
library(rpart)
library(ipred)

set.seed(3060)
bagsizes = c(25, 50, 200, 400, 800)

baggingTrain <-  train.X[,c("label", "nr_pix", "height", "width", "span", "X5rows", "X5cols", "neigh1", "neigh5")]
baggingTrain
test_bag <- bagging (
  formula = label ~ .,
  data = baggingTrain,
  nbagg = 25,
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

1 - test_bag$err

test_bag_cv <- train(
  label ~ .,
  data = baggingTrain,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 5),
  nbagg = 25,
  control = rpart.control(minsplit = 2, cp = 0)
)

test_bag_cv

############ Q2 #############

install.packages("randomForest")
library(randomForest)
library(caret)

baggingTrain

tgrid <- expand.grid(
  mtry = c(2,4,6,8),
  splitrule = "gini",
  min.node.size = 1
)

number_trees <- seq(25, 400, 25)
maxAccuracy = 0
bestTreeFit = 0
bestFit = list()
for (i in 1:16) {
  rf_fit <- train(label ~ ., 
                data = baggingTrain,
                method = "ranger",
                trControl = trainControl(method = "cv", number = 5),
                tuneGrid = tgrid,
                num.trees = number_trees[i]
                )
  if (max(rf_fit$results[4]) > maxAccuracy) {
    maxAccuracy = max(rf_fit$results[4])
    bestTreeFit = i
    bestFit = rf_fit
  } 
}

bestTreeFit
number_trees[bestTreeFit]
bestFit

############ Q3 #############

install.packages("rlist")
library(rlist)
tgrid <- expand.grid(
  mtry = 2,
  splitrule = "gini",
  min.node.size = 1
)

accuracyList = list()
accuricies = 0
for (i in 1:21) {
  rf_fit <- train(label ~ ., 
                  data = baggingTrain,
                  method = "ranger",
                  trControl = trainControl(method = "cv", number = 5),
                  tuneGrid = tgrid,
                  num.trees = 200
  )
  accuracyList[[i]] <- rf_fit$results[4]
  accuricies = accuricies + rf_fit$results[4]
}

accuricies = accuricies / 20
mean(accuracyList)
accuricies
accuracyList
sdAccuracies = 0

for (i in 1:length(accuracyList)) {
  accuracyList[[i]] <- (accuracyList[[i]] - accuricies) ^ 2
  sdAccuracies = sdAccuracies + accuracyList[[i]]
}

sqrt(sdAccuracies / 20)


############ Q4 #############

baggingTrain <- baggingTrain[,c("label", "nr_pix", "width", "span", "X5rows", "X5cols", "neigh1", "neigh5")]

tgrid <- expand.grid(
  mtry = 2,
  splitrule = "gini",
  min.node.size = 1
)

accuracyList = list()
accuricies = 0
for (i in 1:21) {
  rf_fit <- train(label ~ ., 
                  data = baggingTrain,
                  method = "ranger",
                  trControl = trainControl(method = "cv", number = 5),
                  tuneGrid = tgrid,
                  num.trees = 200
  )
  accuracyList[[i]] <- rf_fit$results[4]
  accuricies = accuricies + rf_fit$results[4]
}

accuricies = accuricies / 21
accuricies
accuracyList

sdAccuracies = 0

for (i in 1:length(accuracyList)) {
  accuracyList[[i]] <- (accuracyList[[i]] - accuricies) ^ 2
  sdAccuracies = sdAccuracies + accuracyList[[i]]
}

sqrt(sdAccuracies / 20)







################### DELETE ##########################

kfoldsk = 5

train.X

folds <- cut(seq(1,nrow(train.X)),breaks=kfoldsk,labels=FALSE)
train.X <- cbind(train.X, folds)
tail(train.X$folds)
tail(train.X)

knnk = 1
as.data.frame(train.X)

install.packages('e1071', dependencies=TRUE)
avgAcc = 0
accuracyDF = data.frame(matrix(ncol = 2, nrow = 0))

fs = c('nr_pix', 'height', 'width', 'span', 'X5rows', 'X5cols', 'neigh1', 'neigh5')
for (j in seq(1, 59, 2)) {
  avgAcc = 0
  for(i in 1:kfoldsk) {
    train_items_this_fold  = subset(train.X,train.X$folds != i) 
    validation_items_this_fold = subset(train.X,train.X$folds == i)
    
    # fit knn model on this fold
    predictions = knn(train_items_this_fold[,fs], 
                      validation_items_this_fold[,fs],
                      train_items_this_fold$label, k=j)
    
    correct_list = predictions == validation_items_this_fold$label
    nr_correct = nrow(validation_items_this_fold[correct_list,])
    
    acc_rate = nr_correct/nrow(validation_items_this_fold)
    avgAcc = avgAcc + acc_rate
    print(acc_rate)
  }
  avgAcc = avgAcc / 5
  accuracyDF <- rbind(accuracyDF, cbind(j, avgAcc))
  #save highest acc_rate predictions and items in fold for Q3 confusion matrix
  print(confusionMatrix(predictions, validation_items_this_fold$label))
  
}











