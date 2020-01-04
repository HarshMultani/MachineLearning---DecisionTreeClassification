# Decision Tree Classification
install.packages('caTools')
library(caTools)
install.packages('rpart')
library(rpart)

# Import the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]


# Split the dataset into train and test sets
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Fit the Decision Tree model here
model = rpart(formula = Purchased ~ ., data = training_set, method = 'class')


# Predicting the test set results
Y_pred = predict(model, newdata=test_set[-3] , type = 'class')


# Making the confusion matrix
cm = table(test_set[,3], Y_pred)



# Plotting the Decision Tree
plot(model)
text(model)



