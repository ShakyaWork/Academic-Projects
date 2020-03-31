#Importing the data
file_loc <- "C:/Personal/Data Mining/Census"
setwd(file_loc)

# Import Data
library(data.table)
census_train <- fread("train.csv",na.strings = c(""," ","?","NA",NA)) 
census_tst <- fread("test.csv",na.strings = c(""," ","?","NA",NA)) 

#Viewing data
#View(census_train)
#View(census_tst)


#View its class
#class(census_train)
#class(census_tst)

#Dimensions
#dim(census_train)
#dim(census_tst)

#Column Names
#names(census_train)

#Structure
str(census_train)

#loading dpylr
library(dplyr)
glimpse(census_train)

#Summary
summary(census_train)

#Checking target variable
str(census_train$income_level) #This is an integer 
str(census_tst$income_level) #This is a character
unique(census_train$income_level) #-50000 and 50000
unique(census_tst$income_level) #-50000 and "+50000."

#Replacing these levels with 0 and 1
census_train[,income_level := ifelse(income_level == "-50000",0,1)]
census_tst[,income_level := ifelse(income_level == "-50000",0,1)]

#Confirming replacement
unique(census_train$income_level)
unique(census_tst$income_level)

#Checking if dataset is imbalanced
round(prop.table(table(census_train$income_level))*100)
#Result shows highly imbalanced data

#Updating column classes using data.table package
#Set column classes
factcols <- c(2:5,7,8:16,20:29,31:38,40,41)
numcols <- setdiff(1:40,factcols)

census_train[,(factcols) := lapply(.SD, factor), .SDcols = factcols]
census_train[,(numcols) := lapply(.SD, as.numeric), .SDcols = numcols]

census_tst[,(factcols) := lapply(.SD, factor), .SDcols = factcols]
census_tst[,(numcols) := lapply(.SD, as.numeric), .SDcols = numcols]

#Subsetting Categorical Variables
cat_train <- census_train[,factcols, with=FALSE]
cat_tst <- census_tst[,factcols, with=FALSE]

#Subsetting Numerical Variables
num_train <- census_train[,numcols,with=FALSE]
num_tst <- census_tst[,numcols, with=FALSE]

str(num_train)

#Removing census_train and census_test
#rm(census_train,census_tst)

############################## Data Analysis ###################################
library(ggplot2)
library(plotly)

#Chi-Sq Test to check the relationships of categorical variables
df_class <- table(cat_train$class_of_worker,cat_train$income_level)
chisq_class <- chisq.test(df_class,correct = F)
c(chisq_class$statistic, chisq_class$p.value) 
#p-valueis 0 thus the two variables are correlated
#Similarly we check for rest of the categorical variables

#Males and females
ggplot(data=cat_train, aes(x = sex )) +
  geom_bar(position = "dodge")

#Count of gender by income level
ggplot(data=cat_train, aes(x = sex,fill=income_level )) +
  geom_bar(position = "dodge")

#Writing a function for plots
eda_plot <- function(p) {ggplot(data = num_train, 
                                aes(x = p, y=..density..)) +
    geom_histogram(fill="blue", color="white",alpha=0.6, bins = 100) +
    geom_density()
  ggplotly()
}
#Age
names(num_train)

eda_plot(num_train$age)
eda_plot(num_train$capital_gains)
eda_plot(num_train$capital_losses)

#Adding income level and sex to num_train
num_train[,income_level := cat_train$income_level]
num_train[,sex := cat_train$sex]
names(num_train)

#Wage per hour by age and gender 
ggplot(data = num_train,
       aes(x=age, y=wage_per_hour,color=sex)) + 
  geom_point() +
  ggtitle("Wage per Hour") +
  xlab("Age") + ylab("Wage per hour") +
  theme(plot.title = element_text(hjust=0.5))

#Wage per hour by age and income level 
ggplot(data = num_train,
       aes(x=age, y=wage_per_hour,color=income_level)) + 
  geom_point() +
  ggtitle("Wage per Hour") +
  xlab("Age") + ylab("Wage per hour") +
  theme(plot.title = element_text(hjust=0.5))

names(num_train)
names(cat_train)

#


#Bar graph
ggplot(data=cat_train, aes(x = class_of_worker, fill=income_level )) +
  geom_bar(position = "dodge") +
  theme(axis.text.x =element_text(angle  = 60,hjust = 1,size=10))

#class of workers having income level 1 by gender
ggplot(data=cat_train[income_level==1], aes(x = class_of_worker,fill=sex )) +
  geom_bar() +
  theme(axis.text.x =element_text(angle  = 60,hjust = 1,size=10))

# By Education and income level
ggplot(data=cat_train, aes(x = education, fill=income_level )) +
  geom_bar(position = "dodge") +
  theme(axis.text.x =element_text(angle  = 60,hjust = 1,size=10))

#Income level = 1 by education
ggplot(data=cat_train[income_level==1], aes(x = education )) +
  geom_bar() +
  theme(axis.text.x =element_text(angle  = 60,hjust = 1,size=10))

############################ Data Cleaning #####################################
#Checking for missing values in numerical columns
table(is.na(num_train))
table(is.na(num_tst))

#Finding correlation
library(caret)
library(corrplot)
str(num_train)
find.cor <- cor(num_train[,c(1:7)])
corrplot(find.cor, method = "color",tl.cex = 0.6, order = "alphabet", addCoef.col = "red")
rc <- findCorrelation(x = cor(num_train[,c(1:7)]),cutoff = 0.7)
names(num_train[,7])
num_train <- num_train[,-rc,with=FALSE]
names(num_train) #weeks_worked_in_year is removed as its corr value was more than 0.7

#remove this column from test data
num_tst$weeks_worked_in_year <- NULL
names(num_tst) #weeks_worked_in_year is removed

#Finding missing values in categorical columns
miss_trn <- sapply(cat_train, function(x) {sum(is.na(x))/length(x)})*100
miss_tst <- sapply(cat_tst, function(x) {sum(is.na(x))/length(x)})*100
dfm <- as.data.frame(miss_trn[miss_trn>10])
dfr <- as.data.frame(miss_trn[miss_trn<10&miss_trn>0])
#migration_msa                       49.96717
#migration_reg                       49.96717
#migration_within_reg                49.96717
#migration_sunbelt                   49.96717
#These columns do hold valuable information, thus can be removed
cat_train <- subset(cat_train,select= miss_trn<10)
cat_tst <- subset(cat_tst, select = miss_tst<10)

missval <- cat_train[complete.cases(cat_train)==FALSE,]

write.csv(missval, file="Missing Values.csv")

summary(cat_train)

#set NA as Unavailable - train data
#convert to characters
cat_train <- cat_train[,names(cat_train) := lapply(.SD, as.character),.SDcols = names(cat_train)]

for (i in seq_along(cat_train)) set(cat_train, i=which(is.na(cat_train[[i]])), j=i, value="Unavailable")

#convert back to factors
cat_train <- cat_train[, names(cat_train) := lapply(.SD,factor), .SDcols = names(cat_train)]

#set NA as Unavailable - test data
cat_tst <- cat_tst[, (names(cat_tst)) := lapply(.SD, as.character), .SDcols = names(cat_tst)]

for (i in seq_along(cat_tst)) set(cat_tst, i=which(is.na(cat_tst[[i]])), j=i, value="Unavailable")

#convert back to factors
cat_tst <- cat_tst[, (names(cat_tst)) := lapply(.SD, factor), .SDcols = names(cat_tst)]

#Write it to csv
all.train.data <- cbind(cat_train,num_train)
write.csv(all.train.data,file = "Training Cleaned Data.csv")

#Hygiene Check: Checking if both traaining and test dataset has same levels
library("mlr")
summarizeColumns(cat_train)[,c(10)]
summarizeColumns(cat_tst)[,c(10)]

num_train[,.N,age][order(age)]
num_train[,.N,wage_per_hour][order(-N)]

#Binning the age variable
num_train[,age := cut(x = age, breaks = c(0,30,60,90), include.lowest = TRUE, labels = c("Young","Adult","Old"))]
num_train[,age := as.factor(age)]

num_tst[,age := cut(x = age, breaks = c(0,30,60,90), include.lowest = TRUE, labels = c("Young","Adult","Old"))]
num_tst[,age := as.factor(age)]

num_train[,wage_per_hour := ifelse(wage_per_hour == 0,"Zero","MoreThanZero")][,wage_per_hour := as.factor(wage_per_hour)]
num_train[,capital_gains := ifelse(capital_gains == 0,"Zero","MoreThanZero")][,capital_gains := as.factor(capital_gains)]
num_train[,capital_losses := ifelse(capital_losses == 0,"Zero","MoreThanZero")][,capital_losses := as.factor(capital_losses)]
num_train[,dividend_from_Stocks := ifelse(dividend_from_Stocks == 0,"Zero","MoreThanZero")][,dividend_from_Stocks := as.factor(dividend_from_Stocks)]

num_tst[,wage_per_hour := ifelse(wage_per_hour == 0,"Zero","MoreThanZero")][,wage_per_hour := as.factor(wage_per_hour)]
num_tst[,capital_gains := ifelse(capital_gains == 0,"Zero","MoreThanZero")][,capital_gains := as.factor(capital_gains)]
num_tst[,capital_losses := ifelse(capital_losses == 0,"Zero","MoreThanZero")][,capital_losses := as.factor(capital_losses)]
num_tst[,dividend_from_Stocks := ifelse(dividend_from_Stocks == 0,"Zero","MoreThanZero")][,dividend_from_Stocks := as.factor(dividend_from_Stocks)]

#Removing dependent variable from num_train
num_train[,income_level := NULL]
num_train[,sex := NULL]

#Combining the datasets
d_train <- cbind(cat_train, num_train)
d_tst <- cbind(cat_tst, num_tst)
d_train <- as.data.frame(d_train)
d_tst <- as.data.frame(d_tst)
class(d_train)
#rm(cat_train, num_train,num_train, num_tst)

#Write it to csv
write.csv(d_train, file = "Updated Cleaned Dataset.csv")
levels(d_train$reason_for_unemployment)
############################################################
ggplot(data = d_train, 
       aes(x = age)) +
  geom_bar(fill="blue", color="white",alpha=0.6, bins = 100) 

ggplot(data = d_train, 
       aes(x = d_train$wage_per_hour)) +
  geom_bar(fill="blue", color="white",alpha=0.6, bins = 100)

ggplot(data = d_train, 
       aes(x = d_train$capital_gains)) +
  geom_bar(fill="blue", color="white",alpha=0.6, bins = 100)

ggplot(data = d_train, 
       aes(x = d_train$capital_losses)) +
  geom_bar(fill="blue", color="white",alpha=0.6, bins = 100)


############################################################

#Logistic Linear Model
glm_model <- glm(income_level~.,family = binomial(link="logit"),data = d_train)
summary(glm_model)
#AIC=3469529

library(car)
vif(glm_model)# There are alias coeff.

#the linearly dependent variables
ld.vars <- attributes(alias(glm_model)$Complete)$dimnames[[1]]
alias.cen <- attributes(alias(glm_model)$Complete)
alias.cen$dimnames[[1]]
formula <- as.formula(income_level ~ . - occupation_code 
                      -major_industry_code 
                      -major_occupation_code                        
                      -reason_for_unemployment                       
                      -state_of_previous_residence              
                      -live_1_year_ago              
                      -family_members_under_18            
                      -citizenship )

#remove the linearly dependent variables variables
glm_model.opt <- glm(formula,data = d_train, family = binomial(link = "logit"))

#Anova to test the significance of explanatory variables in the model
anova.test <- anova(glm_model.opt,test = "lrt")
anova.test #Test shows that "d_household_summary", "veterans_benefits" 
          #are insignificant variables. Thus, remove them
formula.new <- as.formula(paste(formula,"-veterans_benefits - d_household_summary"))

#Appying glm again on clean data
glm.census.clean1 <- glm(income_level ~ . - occupation_code -industry_code
                        - major_industry_code - major_occupation_code - 
                          reason_for_unemployment - state_of_previous_residence - live_1_year_ago - 
                          family_members_under_18 - citizenship -veterans_benefits - d_household_summary,
                        data = d_train, family = binomial(link = "logit"))


#VIF
car::vif(glm.census.clean)
#Threshold for VIF is 5. Where as the value of GVIF should be at max sqroot(5)=2.24
#Here we see GIF of year is more than 2.24, thus we should eleminate this variable
#Thus there exists collinearity

glm.census.clean <- glm(income_level ~ . - occupation_code -industry_code
                        - major_industry_code - major_occupation_code - 
                          reason_for_unemployment - state_of_previous_residence - 
                          live_1_year_ago - family_members_under_18 - 
                          citizenship -veterans_benefits - d_household_summary - year,
                        data = d_train, family = binomial(link = "logit"))

summary(glm.census.clean)
#AIC = 53793

#Prediction Using Logistic Linear Model
glm_pred <- predict(glm_model, newdata = d_tst, type = "response")
glm_pred <- as.data.frame(glm_pred)
colnames(glm_pred) <- "Predicted_Level"

#2nd model
glm_pred_2 <- predict(glm_model.opt, newdata = d_tst, type = "response")
glm_pred_2 <- as.data.frame(glm_pred_2)
colnames(glm_pred_2) <- "Predicted_Level"

#3rd Model
glm_pred_3 <- predict(glm.census.clean, newdata = d_tst, type = "response")
glm_pred_3 <- as.data.frame(glm_pred_3)
colnames(glm_pred_3) <- "Predicted_Level"

glm_pred$Predicted_Level <- as.factor(ifelse(glm_pred$Predicted>0.5,1,0))
glm_pred_2$Predicted_Level <- as.factor(ifelse(glm_pred_2$Predicted>0.5,1,0))
glm_pred_3$Predicted_Level <- as.factor(ifelse(glm_pred_3$Predicted>0.5,1,0))

unique(glm_pred)
unique(glm_pred_2)
unique(glm_pred_3)

#Checking Accuracy
library(caret)
confusionMatrix(glm_pred$Predicted_Level,d_tst$income_level)
#72% Accuracy
#Sensitivity : 0.73805         
#Specificity : 0.38021 

confusionMatrix(glm_pred_2$Predicted_Level,d_tst$income_level)
#94.87% Accuracy
#Sensitivity : 0.9890          
#Specificity : 0.3387

confusionMatrix(glm_pred_3$Predicted_Level,d_tst$income_level)
#94.81% Accuracy
#Sensitivity : 0.9891          
#Specificity : 0.3282

############################################################
#Other method of checking Collinearity
findLinearCombos(glm.census.clean$R) #No Collinearity

#Identifying the importnat cateforical variables
library("GoodmanKruskal")
GKmatrix <- GKtauDataframe(d_train)
plot(GKmatrix) #Cannot figure out properly
write.csv(GKmatrix,file = "GKmatrix.csv")
#############################################################

#Removing insignificant columns
rm.col <- c(3,7,8,13,17,18,20,21,25,28,29)
clean_train <- d_train[,-rm.col]
names(clean_train)

#Applying glm on new dataset
glm.final <- glm(income_level~.,family = binomial(link="logit"),data=clean_train)
summary(glm.final)
#AIC: 53048

pred.glm1 <- predict(glm.final, newdata = d_tst, type = "response")
pred.glm1 <- as.data.frame(pred.glm1)
colnames(pred.glm1) <- "Predicted_Level"

pred.glm1$Predicted_Level <- as.factor(ifelse(pred.glm1$Predicted>0.5,1,0))

#Confusion Matrix
confusionMatrix(glm_pred_3$Predicted_Level,d_tst$income_level)
#Accuracy : 0.9481 
#Sensitivity : 0.9891          
#Specificity : 0.3282

#Balancing the dataset
#SMOTE
library(DMwR) #This has smote function
SMOTEdata<-SMOTE(income_level~.,clean_train,perc.over=100,perc_under=200)
table(SMOTEdata$income_level)

SMOTEtst<-SMOTE(income_level~.,d_tst,perc.over=100,perc_under=200)
table(SMOTEtst$income_level)


#Applying glm on new dataset
glm.final <- glm(income_level~.,family = binomial(link="logit"),data=SMOTEdata)
summary(glm.final)
#AIC=30634

pred.glm <- predict(glm.final, newdata = SMOTEtst, type = "response")
pred.glm <- as.data.frame(pred.glm)
colnames(pred.glm) <- "Predicted_Level"

pred.glm$Predicted_Level <- as.factor(ifelse(pred.glm$Predicted>0.5,1,0))

#Confusion Matrix
library(caret)
confusionMatrix(pred.glm$Predicted_Level,SMOTEtst$income_level)
unique(SMOTEtst$income_level)
#Accuracy : 0.8577
#Sensitivity : 0.8502         
#Specificity : 0.8651

#############################################################
#Feature Selection
library(mlr)
task_train <- makeClassifTask(id = deparse(substitute(d_train))
                              ,data = d_train, target = "income_level" )

task_tst <- makeClassifTask(id = deparse(substitute(d_tst))
                              ,data = d_tst, target = "income_level" )

task_train <- removeConstantFeatures(task_train)
task_tst <- removeConstantFeatures(task_tst)

#Balancing imbalanced dataset
#undersampling 
train.under <- undersample(task_train,rate = 0.1) #keep only 10% of majority class
table(getTaskTargets(task_train))

#oversampling
train.over <- oversample(task_train,rate=15) #make minority class 15 times
table(getTaskTargets(train.over))

#SMOTE
train.smote <- smote(task_train,rate = 10,nn = 3)
table(getTaskTargets(train.smote))

#Classification
#naive Bayes
naive_learner <- makeLearner("classif.naiveBayes",predict.type = "response")

#10fold CV - stratified
folds <- makeResampleDesc("CV",iters=10,stratify = TRUE)

#Cross Validation function
func_cv <- function(a) {
  crv_val <- resample(naive_learner,a,folds,
                      measures = list(acc,tpr,tnr,fpr,fp,fn))
crv_val$aggr
}

#Unbalanced Data
func_cv(task_train) #tpr.test.mean=0.7725191,tnr.test.mean=0.8861241

#Undersampled Data
func_cv(train.under) #tpr.test.mean=0.7198356,tnr.test.mean=0.9066387

#Oversampled Data
func_cv(train.over) #tpr.test.mean=0.7091284,tnr.test.mean=0.9093631

#Smote Data
func_cv(train.smote) #tpr.test.mean=0.7324637,tnr.test.mean=0.9073413

#The true positive and true negative accuracy is best for SMOTE, let's use SMOTE data 
#For modeling and prediction

#train and predict
nB_model <- train(naive_learner, train.smote)
nB_predict <- predict(nB_model,task_tst)

#evaluate
nB_prediction <- nB_predict$data$response
dCM <- confusionMatrix(d_tst$income_level,nB_prediction)
dCM 
#Accuracy: 74%
#Sensitivity : 0.9914          
#Specificity : 0.1823

#calculate F measure
precision <- dCM$byClass['Pos Pred Value']
recall <- dCM$byClass['Sensitivity']

f_measure <- 2*((precision*recall)/(precision+recall))
f_measure 

#xgboost
set.seed(2002)
xgb_learner <- makeLearner("classif.xgboost",predict.type = "response")
xgb_learner$par.vals <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  nrounds = 150,
  print.every.n = 50
)

#define hyperparameters for tuning
xg_ps <- makeParamSet( 
  makeIntegerParam("max_depth",lower=3,upper=10),
  makeNumericParam("lambda",lower=0.05,upper=0.5),
  makeNumericParam("eta", lower = 0.01, upper = 0.5),
  makeNumericParam("subsample", lower = 0.50, upper = 1),
  makeNumericParam("min_child_weight",lower=2,upper=10),
  makeNumericParam("colsample_bytree",lower = 0.50,upper = 0.80)
)

#define search function
rancontrol <- makeTuneControlRandom(maxit = 5L) #do 5 iterations

#5 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 5L,stratify = TRUE)
dummyTask<-createDummyFeatures(task_train, target = character(0L), method = "1-of-n",
                               cols = NULL)
#tune parameters
xgb_tune <- tuneParams(learner = xgb_learner, 
                       task = dummyTask, resampling = set_cv, 
                       measures = list(acc,tpr,tnr,fpr,fp,fn), 
                       par.set = xg_ps, control = rancontrol)
# Tune result:
# Op. pars: max_depth=8; lambda=0.198; eta=0.184; subsample=0.837; 
# min_child_weight=8.12; colsample_bytree=0.774 : 
# acc.test.mean=0.9512537, tpr.test.mean=0.9872503,tnr.test.mean=0.4072035,
# fpr.test.mean=0.5927965, fp.test.mean=1468.0000000,fn.test.mean=477.2000000

#global_xgb_tune <<- xgb_tune

#set optimal parameters
xgb_new <- setHyperPars(learner = xgb_learner, par.vals = xgb_tune$x)

#train model
xgmodel <- train(xgb_new, dummyTask)

#test model
predict.xg <- predict(xgmodel, dummyTask)

#make prediction
xg_prediction <- predict.xg$data$response

#make confusion matrix
xg_confused <- confusionMatrix(d_tst$income_level,xg_prediction)
#Accuracy : 0.948
#Sensitivity : 0.9574
#Specificity : 0.6585

precision <- xg_confused$byClass['Pos Pred Value']
recall <- xg_confused$byClass['Sensitivity']

f_measure <- 2*((precision*recall)/(precision+recall))
f_measure
#0.9726374 

