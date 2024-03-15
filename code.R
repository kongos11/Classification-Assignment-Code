# data import
library(tidyverse)
HFdataset <- read.csv('heart_failure.csv')
# View summary
summary(HFdataset)

##########################################
# Global Settings

# Set random seed
set.seed(123)

#Set whether SMOTE is turned on, 1 means resampling starts, 0 means no resampling
# After resampling, the effect is much better, but not resampling is also part of the trial, so keep this switch.
SMOTE_enabled <- 1

##########################################



# Preliminary data analysis, descriptive analysis and visualization
# Draw charts in batches
library(tidyverse)

# Define variable list
varlist <- c("age", "creatinine_phosphokinase", "ejection_fraction",
             "platelets", "serum_creatinine", "serum_sodium", "time")


#Loop through each variable and draw histograms and boxplots for each variable
for (vari in varlist) {
  # Calculate statistics
  MinValues <- min(HFdataset[[vari]], na.rm = TRUE)
  MaxValues <- max(HFdataset[[vari]], na.rm = TRUE)
  MedValues <- median(HFdataset[[vari]], na.rm = TRUE)
  varValues <- var(HFdataset[[vari]], na.rm = TRUE)
  SDValues <- sd(HFdataset[[vari]], na.rm = TRUE)
  
  # Format and display basic information on the title
  HistTitleInfo <- sprintf("Histogram: %s\nMinimum value: %.2f Maximum value: %.2f\n
                            Median: %.2f Variance: %.2f Standard deviation: %.2f",
                           vari, MinValues, MaxValues, MedValues, varValues, SDValues)
  
  # Draw histogram
  HistPlot <- ggplot(HFdataset, aes_string(x = vari)) +
    geom_histogram(fill = "skyblue", color = "black", binwidth = diff(range(HFdataset[[vari]], na.rm = TRUE))/30) +
    theme_minimal() +
    labs(title = HistTitleInfo, x = vari, y = "Frequency")
  print(HistPlot)
  
  # Draw box plot
  BoxTitleInfo <- sprintf("Box plot: %s\nMinimum value: %.2f Maximum value: %.2f\n
                           Median: %.2f Variance: %.2f Standard deviation: %.2f",
                          vari, MinValues, MaxValues, MedValues, varValues, SDValues)
  BoxPlot <- ggplot(HFdataset, aes_string(x = factor(1), y = vari)) +
    geom_boxplot(fill = "tomato", color = "black") +
    theme_minimal() +
    labs(title = BoxTitleInfo, x = "", y = vari)
  print(BoxPlot)
}




# Preliminary data analysis, counting the number of positive examples and negative examples, and drawing correlation analysis diagrams between features
data <- data.frame(
  PNClass = c("Positive", "Negative"),
  Nums = c(sum(HFdataset$fatal_mi == 1), sum(HFdataset$fatal_mi == 0))
)

ggplot(data, aes(x = PNClass, y = Nums, fill = PNClass)) +
  geom_bar(stat = "identity") +
  xlab("Classification") +
  ylab("Total Number") +
  ggtitle("Number of positive examples and negative examples chart")

HFdataset %>% 
  cor() %>% 
  corrplot::corrplot(method = 'number', type = 'upper',order="hclust")




# Preliminary feature engineering, paying special attention to serum_sodium and serum_creatinine

# Draw histograms of the two and add kernel smoothing curves
ggplot(HFdataset, aes(x = serum_sodium)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "lightblue", color = "black") +
  geom_density(color = "red", size = 1) +
  labs(x = "serum_sodium", y = "Density", title = "serum_sodium histogram with kernel smooth curve") +
  theme_minimal()

# Create a combined histogram and kernel smooth plot
ggplot(HFdataset, aes(x = serum_creatinine)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "lightblue", color = "black") +
  geom_density(color = "red", size = 1) +
  labs(x = "serum_sodium", y = "Density", title = "serum_sodium histogram with kernel smooth curve") +
  theme_minimal()



# Set the upper and lower bound thresholds of serum_creatinine normal indicators
SCLower <- 0.6
SCUpper <- 1.35

#Create serum_creatinine_point feature
HFdataset$serum_creatinine_point <- ifelse(HFdataset$serum_creatinine >= SCLower & HFdataset$serum_creatinine <= SCUpper, 1, 0)

# Set the upper and lower thresholds of the normal indicator of serum_sodium
SSLower <- 135
SSUpper <- 145

#Create serum_sodium_point feature
HFdataset$serum_sodium_point <- ifelse(HFdataset$serum_sodium >= SSLower & HFdataset$serum_sodium <= SSUpper, 1, 0)

# Find the index of fatal_mi column
fatalmiIndex <- which(colnames(HFdataset) == "fatal_mi")

# Move the fatal_mi column to the last column
HFdataset <- HFdataset[, c(1:(fatalmiIndex-1), (fatalmiIndex+1):ncol(HFdataset), fatalmiIndex)]

# head(HFdataset)



#Load caret library
library(caret)

# Determine whether to perform SMOTE resampling
if (SMOTE_enabled == 1) {
  
  #Load the libraries required for SMOTE
  library(smotefamily)
  #Perform SMOTE resampling
  oversampled_data <- SMOTE(X = HFdataset[, -ncol(HFdataset)], target = HFdataset[, ncol(HFdataset)])
  oversampled_data <- oversampled_data$data
  colnames(oversampled_data)[ncol(oversampled_data)] = 'fatal_mi'
  HFdataset <- oversampled_data
}




# Divide training set and test set, 70% training set, 30% test set
DataIndex <- sample(1:nrow(HFdataset), nrow(HFdataset) * 0.7)
TrainData <- HFdataset[DataIndex,]
TrainData$fatal_mi <- factor(TrainData$fatal_mi, levels = c(0, 1))
TestData <- HFdataset[-DataIndex,]

#Training set features
n_row <- ncol(HFdataset)
TrainFeature <- as.matrix(TrainData[,1:n_row-1])
TrainFeature <- scale(TrainFeature)

#Training set labels
TrainLabel <- TrainData[,n_row]

#Test set features
TestFeature <- as.matrix(TestData[,1:n_row-1])
TestFeature <- scale(TestFeature)

#Training set labels
TestLabel <- TestData[,n_row]





# Preliminary classification model exploration, using KNN model, only one feature engineering was performed before

# According to the actual situation, define the evaluation function of recall rate
RecallSummary <- function(data, lev = NULL, model = NULL) {
  recall <- sensitivity(data[, "pred"], data[, "obs"], lev[1])
  names(recall) <- "Recall"
  recall
}

#Set training control parameters and use recall rate as the evaluation indicator
KNNControl <- trainControl(method = "cv",
                           number = 5,
                           summaryFunction = RecallSummary)

# Search K for 3 5 7 9
KNNGrid <- expand.grid(.k = c(3, 5, 7, 9))

#Train KNN model
KNNmodel <- train(x = TrainFeature,
                  y = TrainLabel,
                  method = "knn",
                  trControl = KNNControl,
                  tuneGrid = KNNGrid)

# Use test set for prediction
predictions <- predict(KNNmodel, newdata = TestFeature)

# Create a data frame containing predictions and observations
KNNResult <- data.frame(pred = predictions, obs = TestLabel)
ConfusionMatrix <- table(KNNResult)

# Output the optimal k value
print(paste("Optimal k:", KNNmodel$bestTune$.k))

print(ConfusionMatrix)
# Calculate Precision
precision <- ConfusionMatrix[2, 2] / sum(ConfusionMatrix[2, ])

# Calculate Specificity
specificity <- ConfusionMatrix[1, 1] / sum(ConfusionMatrix[1, ])

# Calculate Accuracy
accuracy <- (ConfusionMatrix[1, 1] + ConfusionMatrix[2, 2]) / sum(ConfusionMatrix)

#Print evaluation indicators
print(paste("Recall:", ConfusionMatrix[2,2] / sum(ConfusionMatrix[,2])))
print(paste("Precision:", precision))
print(paste("Specificity:", specificity))
print(paste("Accuracy:", accuracy))


# Convert data format
ConfusionDF <- as.data.frame(as.table(ConfusionMatrix))


# Draw confusion matrix heat map
ggplot(ConfusionDF, aes(x = pred, y = obs, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1.5, color = "white", size = 6) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = median(ConfusionDF$Freq)) +
  labs(title = "Confusion Matrix Heatmap", x = "Predicted Results", y = "Actual Results") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5))






# Preliminary random forest, still only the first feature engineering
library(randomForest)

RepeatCV <- trainControl(method='repeatedcv', number = 5, repeats = 3 )

RandForestModel <- train(fatal_mi ~ age + creatinine_phosphokinase + .,
                         data = TrainData,
                         method = "rf",
                         trControl = RepeatCV,
                         metric = "Accuracy")
RandForestModel$finalModel

TestPred <- predict(RandForestModel, TestData)
# Calculate confusion matrix
ConfusionMatrix <- table(TestData$fatal_mi, TestPred)

# Calculate precision (Precision)
precision <- ConfusionMatrix[2, 2] / sum(ConfusionMatrix[2, ])

# Calculate specificity (Specificity)
specificity <- ConfusionMatrix[1, 1] / sum(ConfusionMatrix[1, ])

# Calculate accuracy (Accuracy)
accuracy <- (ConfusionMatrix[1, 1] + ConfusionMatrix[2, 2]) / sum(ConfusionMatrix)

# Calculate recall rate
Recall <- ConfusionMatrix[2, 2] / sum(ConfusionMatrix[2, ])

# Output evaluation indicators
print(paste("Recall:", Recall))
print(paste("Precision:", precision))
print(paste("Specificity:", specificity))
print(paste("Accuracy:", accuracy))


#Convert confusion matrix to data frame
ConfusionDf <- as.data.frame(as.table(ConfusionMatrix))

# Draw heatmap using correct column names
ggplot(data = ConfusionDf, aes(x = Var1, y = TestPred, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1.5, color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = median(ConfusionDf$Freq)) +
  labs(title = "Confusion Matrix Heat Map", x = "Predicted Results", y = "Actual Results") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))



#Feature importance ranking
VarImp <- varImp(RandForestModel, scale=FALSE)$importance
VarImp <- data.frame(variables=row.names(VarImp), importance=VarImp$Overall)

#Set the importance threshold to 5
Selected_features <- VarImp$variables
Selected_indexes <- which(VarImp$importance > 5)

var <- VarImp[VarImp$importance>5,][order(VarImp[VarImp$importance>5,]$importance, decreasing = TRUE),]

VarImp%>%
  
  ## Sort the data by importance
  arrange(importance) %>%
  
  ## Create a ggplot object for aesthetic
  ggplot(aes(x=reorder(variables, importance), y=importance)) +
  
  ## Plot the bar graph
  geom_bar(stat='identity') +
  
  ## Flip the graph to make a horizontal bar plot
  coord_flip() +
  
  xlab('Variables') +
  labs(title='Random forest variable importance') +
  theme_minimal() +
  theme(axis.text = element_text(size = 10),
        axis.title = element_text(size = 15),
        plot.title = element_text(size = 20),
  )



# The second round of feature engineering, delete unimportant features, and update the training set data set
n_row <- ncol(HFdataset)
TrainFeature <- as.matrix(TrainData[,1:n_row-1])
TrainFeature <- scale(TrainFeature)
TrainFeature <- TrainFeature[,Selected_indexes]
TrainLabel <- TrainData[,n_row]

TestFeature <- as.matrix(TestData[,1:n_row-1])
TestFeature <- scale(TestFeature)
TestFeature <- TestFeature[,Selected_indexes]
TestLabel <- TestData[,n_row]




# Random forest training and testing after the second round of feature engineering

forest <- train(x = TrainFeature,
                y = TrainLabel,
                method = "rf",
                trControl = RepeatCV,
                metric = "Accuracy")
forest$finalModel

TestPred <- predict(forest, TestFeature)
# Calculate confusion matrix
ConfusionMatrix <- table(TestLabel, TestPred)

# Calculate precision (Precision)
precision <- ConfusionMatrix[2, 2] / sum(ConfusionMatrix[, 2])

# Calculate specificity (Specificity)
specificity <- ConfusionMatrix[1, 1] / sum(ConfusionMatrix[1, ])

# Calculate accuracy (Accuracy)
accuracy <- (ConfusionMatrix[1, 1] + ConfusionMatrix[2, 2]) / sum(ConfusionMatrix)

# Calculate recall rate
Recall <- ConfusionMatrix[2, 2] / sum(ConfusionMatrix[2, ])

# Output evaluation indicators
print(paste("Recall:", Recall))
print(paste("Precision:", precision))
print(paste("Specificity:", specificity))
print(paste("Accuracy:", accuracy))


#Convert confusion matrix to data frame
ConfusionDf <- as.data.frame(as.table(ConfusionMatrix))
names(ConfusionDf) <- c("Actual", "Predicted", "Frequency")

# Draw heat map
ggplot(data = ConfusionDf, aes(x = Predicted, y = Actual, fill = Frequency)) +
  geom_tile() +
  geom_text(aes(label = Frequency), vjust = 1.5, color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = median(ConfusionDf$Frequency)) +
  labs(title = "Confusion Matrix Heatmap", x = "Predicted Label", y = "Actual Label") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))



# KNN after the second round of feature engineering

#Define a recall evaluation function
recallSummary <- function(data, lev = NULL, model = NULL) {
  recall <- sensitivity(data[, "pred"], data[, "obs"], lev[1])
  names(recall) <- "Recall"
  recall
}

#Set training control parameters and use recall rate as the evaluation indicator
KNNcontrol2 <- trainControl(method = "cv",
                            number = 5,
                            summaryFunction = recallSummary)

# Set the candidate value of K
KNNGrid <- expand.grid(.k = c(3, 5, 7, 9)) # For example, the candidate values of K are 3, 5, 7, 9

#Train KNN model
KNNModel2 <- train(x = TrainFeature,
                   y = TrainLabel,
                   method = "knn",
                   trControl = KNNcontrol2,
                   tuneGrid = KNNGrid)

# Use test set for prediction
predictions <- predict(KNNModel2, newdata = TestFeature)

# Create a data frame containing predictions and observations
KNNResult <- data.frame(pred = predictions, obs = TestLabel)
ConfusionMatrix <- table(KNNResult)

# Output recall rate
ConfusionMatrix
print(paste("Recall:", ConfusionMatrix[2,2]/sum(ConfusionMatrix[,2])))



# MLP after doing all feature engineering
library(keras)
model <- keras_model_sequential()
model%>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)



TrainLabel <- to_categorical(TrainLabel)
TestLabel <- to_categorical(TestLabel)



#Train MLP model
history <- model %>% fit(
  x = TrainFeature,
  y = TrainLabel,
  epochs = 50,
  batch_size = 16,
  verbose=2
)



# Output training results
summary(model)

model %>% evaluate(TestFeature, TestLabel)

plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="orange", type="l")
lines(history$metrics$val_loss, col="skyblue")
legend("topright", c("Training","Testing"), col=c("orange", "skyblue"), lty=c(1,1))

TestPred <- predict(model, TestFeature)
TestPred <- max.col(TestPred)-1





# Calculate confusion matrix
ConfusionMatrix <- table(TestData$fatal_mi, TestPred)


# Calculate recall rate (Recall)
Recall <- ConfusionMatrix[2, 2] / sum(ConfusionMatrix[2, ])

# Calculate precision (Precision)
precision <- ConfusionMatrix[2, 2] / sum(ConfusionMatrix[, 2])

# Calculate specificity (Specificity)
specificity <- ConfusionMatrix[1, 1] / sum(ConfusionMatrix[1, ])

# Calculate accuracy (Accuracy)
accuracy <- (ConfusionMatrix[1, 1] + ConfusionMatrix[2, 2]) / sum(ConfusionMatrix)

# Output evaluation indicators
print(paste("Recall:", Recall))
print(paste("Precision:", precision))
print(paste("Specificity:", specificity))
print(paste("Accuracy:", accuracy))

library(ggplot2)

#Convert confusion matrix to data frame
ConfusionDf <- as.data.frame(as.table(ConfusionMatrix))
names(ConfusionDf) <- c("Actual", "Predicted", "Frequency")

# Draw heat map
ggplot(data = ConfusionDf, aes(x = Predicted, y = Actual, fill = Frequency)) +
  geom_tile() +
  geom_text(aes(label = Frequency), vjust = 1.5, color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = median(ConfusionDf$Frequency)) +
  labs(title = "Confusion Matrix Heatmap", x = "Predicted Label", y = "Actual Label") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))


