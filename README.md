# Give Me Some Credit

## Set 2: [PCA/Hyperparameter/CV] [Due by 3.29 Wed]
  * The goal of this HW is to be familiar with PCA (feature extraction), grid search, pipeline, k-fold CV. 
  * For this HW, we continue to use [Give Me Some Credit]([http://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)) on Kaggle. 
  * Extract a few (>2) features using PCA method.
  * Using the selected features from above, we are going to apply LR / SVM / decision tree (or any other algorithm). 
  * Implement the methods using pipeline. (__PML__ p185)
  * Use grid search for finding optimal hyperparameters. (__PML__ p199). In the search, apply 5-fold cross-validation.
  

### Dealing with data imbalance

The data of credit scoring is not balance between default and no default condition. Thus, to solve this issues, we need to do resampling. In this notebook, the downsampling is used to reduce the time of computation and solve the imbalance issue. Below is the number of data with label 1 and 0 before resampling.
![Data before resampling](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/data_beforeresample_dist.png)
After the downsampling, the total number of data become around 12000.
![Data after downsampling](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/data_downsample_dist.png)

### Optimum hyperparameter using grid search

I used gridsearch to find the optimum parameter for 3 classifier (LR, SVM, Decision tree). Some parameter that was varied was `C`, tree `max depth`, and the tree classifier 'criterion'. I also used the pipeline to choose 3 feature using PCA and then 5-fold cross validation.

The output of the optimum accuracy were as follows:

`
0.6946431264639525
{'logisticregression__C': 10.0}`

`
0.7048177276798568
{'decisiontreeclassifier__criterion': 'entropy', 'decisiontreeclassifier__max_depth': 10.0}`

`\n
0.7153645656263456
{'svc__C': 10.0}`

### Majority Vote Classifier

I apply the optimum hyperparameter to the pipeline and do the majority vote classifier. This classifier composed of the three classifier that previously optimised. The result was shown in terms of area under Receiver Operator Characteristic (ROC) curve.

`
ROC AUC: 0.77 (+/- 0.00418) [Logistic regression]
`

`
ROC AUC: 0.75 (+/- 0.00615) [Decision tree]
`

`
ROC AUC: 0.78 (+/- 0.00497) [SVC]
`

`
ROC AUC: 0.84 (+/- 0.00827) [Majority voting]
`

![ROC AUC](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/ROC_AUC.png)

The TPR vs FPR curve also can be drawn
![TPR vs FPR](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/TPRvsFPR.png)

### Conclusion
Overall, the workflow can be streamlined using pipeline function. The highest performance was achieved by using majority classifier which was 0.84 (ROC AUC). The majority classifier which include the other 3 classifier can increased the accuracy of the model for around 6%. The accuracy of the model may increase through increasing more classifier in the majority class clasifier. However, more computation source is needed to do the task.

### Appendices
Some supplementary figures for illustration during processing the data.

#### PCA cumulative explained variance
![PCA cumulative explained variance](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/pca_cumulative_expvariance.png)

#### Learning Curve
![Learning Curve Martin Adrian ITB Bandung Indonesia](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/learning_curve.png)

#### Validation curve
![Validation curve](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/validation_curve.png)

#### Confusion Matrix
Decision tree confusion matrix
![Confusion Matrix Decision tree](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/confusion_matrix_dt.png)

Logistic regression confusion matrix
![Confusion Matrix Logistic regression](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/confusion_matrix_lr.png)


## Set 1: [Classifiers] [Due by 3.21 Tues]
  * The goal of this HW is to be familiar with the basic classifiers __PML__ Ch 3. 
  * For this HW, we will use [Give Me Some Credit]([http://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)) on Kaggle. You may download from Kaggle link or CMS.
  * Load `cs-training.csv` into a Pandas dataframe.
  * Fill-in the missing values (`nan`) with the column means. (Use `pd.fillna()` or See Ch 4 of `PML`)
  * Select the 2 most important features using LogisticRegression with L1 penalty. (Adjust C until you see 2 features)
  * Using the 2 selected features, apply LR / SVM / decision tree. Try your own hyperparameters (C, gamma, tree depth, etc) to maximize the prediction accuracy. (Just try several values. You don't need to show your answer is the maximum.)
  * Visualize your classifiers using the `plot_decision_regions` function from __PML__ Ch. 3
  * Put your result in `YOUR_GITHUB_ID/Give-Me-Some-Credit/code/Classifiers.ipynb`

## Features weight by varying C values
![Features](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/plot_C2.png)
![Features more C range](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/plot_C.png)

## Logistic Regression
Logistic regression using two different values of C

![LR1](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/lr01.png)

![LR2](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/lr02.png)

## Decision tree
Decision tree using two different values of max_depth of the decision tree

![DT1](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/decision_depth5.png)

![DT2](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/decision_depth1000.png)

## SVM
SVM (linear kernel) using two different values of C

![SVM1](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/svm_lin_c_0001.png)

![SVM2](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/svm_lin_c_10.png)

<!--
<img src="https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/lr01.png" width="50" height="50">
<img src="https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/lr02.png" width="50" height="50">
-->


