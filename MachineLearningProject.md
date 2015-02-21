# Practical Machine Learning Project

###Synopsis

The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. We will use the variables in the training dataset to predict the "classe" outcome variable. The classe varible has five levels: (Class A) - exactly according to the specification, (Class B) - throwing the elbows to the front, (Class C) - lifting the dumbbell only halfway, (Class D) - lowering the dumbbell only halfway, (Class E) - throwing the hips to the front. Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.  Finally, we will use our final prediction model to predict the 20 different test cases. 


###Data

The training data for this project are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv]

The test data are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv]

The data for this project come from this source: [http://groupware.les.inf.puc-rio.br/har].


####Reproducibility

We are using the pseudo-random number generator seed whenever we build a model in order to enhance reproducibility.

####Expected Out Of Sample Error

The expected out of sample error will correspond to one minus the accuracy (1 - accuracy) using the testing dataset. Accuracy is the proportion of correct classified observation over the total sample in the testing dataset. In the literature, the accuacry is approximately 99.4%. Thus, we expect that our expected out of sample error will be between 0.45% to 0.6%.


###Data Processing

Some missing values are coded as string "#DIV/0!" or "" or "NA". These will be changed to NA. Irrelevant variables will be deleted.



```r
# Loading the training dataset and replacing all missing with "NA"
traindat<-read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
# Loading the testing dataset 
testdat<-read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))
# Check dimensions for number of variables and number of observations
dim(traindat)
```

```
## [1] 19622   160
```

```r
dim(testdat)
```

```
## [1]  20 160
```

Some variables are irrelevant to our current project: (columns 1 to 7) ie. user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, and num_window. We will delete these variables.



```r
#Delete columns with all missing values
traindat<-traindat[,colSums(is.na(traindat)) == 0]
testdat<-testdat[,colSums(is.na(testdat)) == 0]
#Removing irrelevant variables.
traindat<-traindat[,-c(1:7)]
testdat <-testdat[,-c(1:7)]
#Check our new datasets:
dim(traindat)
```

```
## [1] 19622    53
```

```r
dim(testdat)
```

```
## [1] 20 53
```

**Partitioning the training dataset to allow cross-validation**

The training dataset contains 53 variables and 19622 observations.
The testing dataset contains 53 variables and 20 observations.
In order to perform cross-validation, the training dataset is partitioned into 2 subsets: subtraining (75%) and subtesting (25%).


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(3523)
intrain <- createDataPartition(y=traindat$classe, p=0.75, list=FALSE)
training <- traindat[intrain, ] 
testing <- traindat[-intrain, ]
```

**First prediction model: Decision Tree**


```r
firstfit<-train(classe~.,method="rpart",data=training)
```

```
## Loading required package: rpart
```

```r
pred1<-predict(firstfit,newdata=testing)
confusionMatrix(pred1,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1272  388  394  360  136
##          B   21  333   29  152  122
##          C   98  228  432  292  241
##          D    0    0    0    0    0
##          E    4    0    0    0  402
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4973          
##                  95% CI : (0.4833, 0.5114)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.343           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9118   0.3509  0.50526   0.0000  0.44617
## Specificity            0.6358   0.9181  0.78785   1.0000  0.99900
## Pos Pred Value         0.4988   0.5068  0.33462      NaN  0.99015
## Neg Pred Value         0.9477   0.8550  0.88292   0.8361  0.88906
## Prevalence             0.2845   0.1935  0.17435   0.1639  0.18373
## Detection Rate         0.2594   0.0679  0.08809   0.0000  0.08197
## Detection Prevalence   0.5200   0.1340  0.26325   0.0000  0.08279
## Balanced Accuracy      0.7738   0.6345  0.64656   0.5000  0.72259
```

The Decision Tree's accuracy is not high enough.


**Second prediction model: Random Forest**


```r
set.seed(33833)
modfit<-train(classe~.,data=training,method="rf",trControl =trainControl(method="cv",number=5))
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
pred2<-predict(modfit,newdata=testing)
confusionMatrix(pred2, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    5    0    0    0
##          B    2  944    1    1    0
##          C    0    0  850    5    2
##          D    0    0    4  798    2
##          E    0    0    0    0  897
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9955          
##                  95% CI : (0.9932, 0.9972)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9943          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9947   0.9942   0.9925   0.9956
## Specificity            0.9986   0.9990   0.9983   0.9985   1.0000
## Pos Pred Value         0.9964   0.9958   0.9918   0.9925   1.0000
## Neg Pred Value         0.9994   0.9987   0.9988   0.9985   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2841   0.1925   0.1733   0.1627   0.1829
## Detection Prevalence   0.2851   0.1933   0.1748   0.1639   0.1829
## Balanced Accuracy      0.9986   0.9969   0.9962   0.9955   0.9978
```


###Conclusion

Random Forest's prediction accuracy is better than that of Decision Tree. The accuracy for Random Forest model is 0.995. Hence, the expected out of sample error is estimated at 0.005, or 0.5%. The Random Forest model is choosen to predict the 20 different test cases.


##Submission


```r
#predict outcome levels on the original Testing data set using Random Forest algorithm
finalsub <- predict(modfit, testdat)
finalsub
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



```r
# Write files for submission
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(finalsub)
```
