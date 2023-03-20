# Give Me Some Credit

### Instructions: Set 1: [Classifiers] [Due by 3.21 Tues]
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
<img src="[https://your-image-url.type](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/lr01.png)" width="50" height="50">
<img src="[https://your-image-url.type](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/lr02.png)" width="50" height="50">
![Features more C range](https://github.com/martinbandung/Give-Me-Some-Credit/blob/main/code/images/lr02.png)
