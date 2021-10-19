# Step 0: Load necessary libraries
############################################################################################################################################################################

#set seed
set.seed(3102)

#load libraries
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(caret)
  library(tidyr)
  library(xgboost)
  library(glmnet)
  library(e1071)
})


# utility functions to perform data curation

dataCurateAndTransform = function(df){
  
  charVars <- colnames(df)[
    which(sapply(df,
                 function(x) mode(x)=="character"))]
  df_train <- df[, !colnames(df) %in% charVars, 
                 drop=FALSE]
  n.train <- nrow(df_train)
  for(var in charVars){
    
    mylevels <- sort(unique(df[, var]))
    m <- length(mylevels)
    m <- ifelse(m>2, m, 1)
    tmp.train <- matrix(0, n.train, m)
    col.names <- NULL
    for(j in 1:m){
      tmp.train[df[, var]==mylevels[j], j] <- 1
      col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
    }
    colnames(tmp.train) <- col.names
    df_train <- cbind(df_train, tmp.train)
  }
  return(df_train)
}

winsorize_test=function(trainData, testData, winsor.vars) {
  quan.value <- 0.95
  for(var in winsor.vars){
    tmp1 <- trainData[, var]
    tmp2 <- testData[, var]
    myquan1 <- quantile(tmp1, probs = quan.value, na.rm = TRUE)
    myquan2 <- quantile(tmp2, probs = quan.value, na.rm = TRUE)
    tmp2[tmp2 > myquan2] <- myquan1
    testData[, var] <- tmp2
  }
  testData
}

ignoreNullsOnGarageYr = function(data) {
  data$Garage_Yr_Blt[is.na(data$Garage_Yr_Blt)] = 1978
  data
}

hotEncoding = function(dataset, testFlag) {
  if(testFlag == TRUE) {
    dummies_model =  dummyVars(" ~ .", data=dataset)
  }
  else {
    dummies_model= dummyVars(Sale_Price ~ ., data=dataset)
  }
  dataset = predict(dummies_model, newdata = dataset)
  dataset = data.frame(dataset)
  return(dataset)
}

RMSE = function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

removeVars = function(data,remove.var) {
  data[,!names(data) %in% remove.var]
}

winsorize = function(data, winsor.vars) {
  
  quan.value <- 0.95
  for(var in winsor.vars){
    tmp <- data[, var]
    myquan <- quantile(tmp, probs = quan.value, na.rm = TRUE)
    tmp[tmp > myquan] <- myquan
    data[, var] <- tmp
  }
  data
}

remove.var <- c('Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude')

winsor.vars <- c("Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val")

#Data Preprocessing 
myPreProcess = function(data,testFlag) {
  
  d <- preProcess(data, "medianImpute")
  data = predict(d,data)
  data = winsorize(data, winsor.vars)
  data_encoded = hotEncoding(data,testFlag)
  data_encoded = removeVars(data_encoded,remove.var)
  data_encoded$PID = data$PID
  if("Sale_Price" %in% colnames(data))
  {
    data_encoded$Sale_Price = data$Sale_Price
    data_encoded$Sale_Price_Log = log(data$Sale_Price)
  }
  data_encoded <- data_encoded %>% 
    fill(
      dplyr::everything()
    )
  data_encoded
}

############################################################################################################################################################################
# Step 1: Preprocess training data
#         and fit two models

# curate loaded train data 
trainData <- read.csv("train.csv")
trainData_XG <- trainData
trainData = trainData[,setdiff(colnames(trainData),remove.var)]
winsorizedTrainData=trainData
trainData=winsorize(trainData, winsor.vars)
trainData=ignoreNullsOnGarageYr(trainData)
trainData = dataCurateAndTransform(trainData)
categorizedTrainData=trainData
trainData=trainData[,sort(colnames(trainData))]

trainData<-trainData[sample(nrow(trainData)),]
X_train = as.matrix(subset(trainData, select=-c(PID,Sale_Price)))
Y_train = trainData$Sale_Price

trainDataXgb = myPreProcess(trainData_XG,FALSE)
fit = lm(Sale_Price ~ Lot_Area + Mas_Vnr_Area , data = trainDataXgb)
fit_cd = cooks.distance(fit)
trainData = trainData[fit_cd < 4 / length(fit_cd),]

######################################################################################
# Step 1a: fit model 1 

lambdas <- 10^seq(2, -3, by = -.1)
set.seed(3102)
ridge_reg = glmnet(X_train, log(Y_train), nlambda = 20, alpha = 0.7, family = 'gaussian',
                   lambda = lambdas,type.measure='mse')
cv_ridge <- cv.glmnet(X_train, log(Y_train), alpha = 0.7, lambda = lambdas,nfolds=10)


######################################################################################
# Step 1b: fit model 2
set.seed(3102)
xgb.model <- xgboost(data = X_train, label = log(Y_train),
                     nrounds = 1000, verbose = FALSE, objective = "reg:squarederror", eval_metric = "rmse", 
                     eta = 0.0474239, max_depth = 4, subsample = 0.541795)

############################################################################################################################################################################
# Step 2: Preprocess test data
#         and output predictions into two files
#
testData <- read.csv("test.csv")
testDataXgb <- testData
testDataXgb = myPreProcess(testDataXgb,TRUE)
testData = testData[,setdiff(colnames(testData),remove.var)]
testData = winsorize_test(winsorizedTrainData, testData, winsor.vars)
testData=ignoreNullsOnGarageYr(testData)
testData = dataCurateAndTransform(testData)
var1=colnames(categorizedTrainData)
var2=colnames(testData)
non.matched_1 = var1[!var1 %in% var2]
to_be_added_test=non.matched_1
non.matched_2 = var2[!var2 %in% var1]
to_be_deleted_test=non.matched_2
testData_tmp = testData[,setdiff(colnames(testData),to_be_deleted_test)]
for(i in to_be_added_test){
  testData_tmp[,i] = 0
}
testData_tmp = testData_tmp[,setdiff(colnames(testData_tmp),"Sale_Price")]
testData=testData_tmp
testData=testData[,sort(colnames(testData))]
X_test = as.matrix(subset(testData, select=-c(PID)))

# ######################################################################################
# # Step 2a: predict model 1 

optimal_lambda <- cv_ridge$lambda.min
predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = X_test)
predictions_test = exp(predictions_test)

output = data.frame(testData$PID)
output$Sale_Price = predictions_test
colnames(output) = c("PID","Sale_Price")
write.csv(output,"mysubmission1.txt",row.names = F)
# 
# ######################################################################################
# # Step 2b: predict model 2
pred <- predict(xgb.model , X_test)
pred_rf = exp(pred)
output2 = data.frame(testData$PID)
output2$Sale_Price = pred_rf
colnames(output2) = c("PID","Sale_Price")  
write.csv(output2,"mysubmission2.txt",row.names = F)