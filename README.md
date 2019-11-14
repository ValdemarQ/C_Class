# C_Class


## **SPRINT 1**

## Intro to Machine Learning | Kaggle | 1 - 5 Modules
- Decision trees - simple explanation how they work (House value predicted decisions tree, based on questions, rooms more than 2?, then this price)
- Importance to familiarise with data using Pandas, as data may be old, missing values,
thus model would not perform well given inacurate, old data on new examples.
- Building DecisionTree model with Scklearn to predict price based on selected features: Rooms, Bathrooms, Long,Lat
- Using Decision tree model to build ML predicting prices of Iowa house prices
- Feature selection & Prediction selection - explaninig how to select features and prediction target
- Model traininig - explaninig how to train model


## ML FAST.AI - LESSON 1

- Sckitlean library is most impornat and popular ML library in python
- Kaggle is great source to learn with competitions
- It's always important to look into your data and understand it
- Regresson - to predict continous data , Classifier - to predict categorical variables
- Regression - is ML model tryign to predict continuous outcome/variable
- Datetime contains a lot of information, good source for Feature engineering, dependent on your target
- Ordered some categorical ordinal data
- No harm in adding extra columns to your dataset (So go ahead)
- Feather - some light and fast way of loading,saving data
- Some great Fastai premade structures (def proc_dc, numeralize, fix_missing etc) - He used datas well, worked with missing values etc., and
got really great results, well at least - 25%



## Machine Learning with Python - Coursera
- ML =  Ability to learn without explicitly programmed.
- Basics of ML, Attributes, features, prediction target etc.
- Supervised and Unsupervised ML 
- What is classification - to predict class
- Regression is to predict continuous data



## **SPRINT 2**

## Kaggle: Intro to Machine Learning - Model Validation - Underfitting and Overfitting

- **Model Validation** - How good is your mode? 

    - One of the possible metrics is Mean Absolute Error.

    - It's important to validate data that model has not seen before.

    - **Train_test_split**, used to break up the data into two pieces (Train and Test)

    - Building simple model to predict house prices with Decision Trees, training & validating

    - Simple practical lesson to predict price with DT and MAE calculation.

- ### Underfitting and Overfitting
    - **Overfitting**: capturing spurious patterns that won't recur in the future, leading to less accurate prediction

    - **Underfitting**: failing to capture relevant patterns, again leading to less accurate predictions.

    - Use validation to measure accuracy

    - Practice, DT model with various max_leaf_nodes to find lowest MAE and use it for model.


## ML FAST.AI - LESSON 2 — RANDOM FOREST DEEP DIVE
- Creating a validation set is the most important thing you need to do when you are doing a machine learning project

 - %time, it will tell you how long things took

 - Building a single tree - with n_estimators=1, max_depth=3, and visualising it, to see how it's done.

- Random forest — a way of bagging trees. if we created five different models each of which was only somewhat predictive but the models gave predictions that were not correlated with each other. average of those five models, you are effectively bringing in the insights from each of them. So this **idea of averaging models** is a technique for Ensembling.

- Purpose of modeling in ML is to find a model which tells you which variables are important and how do they interact together to drive your dependent variable.

- The effective machine learning model is accurate at finding the relationships in the training data and generalizes well to new data 

- Suggestion when starting to build RF models, use less trees, and only later use large - as it takes a lot of computing.

- oob_score=True - would use unused rows through the first tree and treat it as a validation set

- **Grid search**. Scikit-learn  function, where you pass in a list of all the hyper parameters you want to tune and all of the values of these hyper parameters. It will run your model on every possible combination of all these hyper parameters and tell you which one is the best.

- **ERROR -** Most people run all of their models on all of the data all of the time using their best possible parameters which is just pointless.

    - **TIP** Do most of your models on a large enough sample size that your accuracy is reasonable (within a reasonable distance of the best accuracy you can get) and taking a small number of seconds to train so that you can interactively do your analysis.

- **min_samples_leaf=3** - Stop training the tree further when a leaf node has 3 or less samples. We are taking the average of at least three points that we would expect the each tree to generalize better. (Recommended values: 1,3,5,10,25)

- **max_features=0.5** - So if every tree always splits on the same thing the first time, you will not get much variation in those trees. 
  - This param, indicates to every split from different subset of columnts. 
  - 0.5 means randomly choose a half of them.
  - Good values to use are 1, 0.5, log2, or sqrt

- **Why Random Forest works so well?** With just two splits, we can pull out a single category. Tree is infinitely flexible even with a categorical variable