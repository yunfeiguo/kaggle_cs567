#!/bin/bash
set -e
#Feature generation
./cleanAndAggByMean.py #clean and aggregate data, add meteorologists' features
./addNewFeature.py #add some new features
#Model evaluation
./lasso_CV.py -f Zdr #run CV on lasso using Zdr as feature
./elasticNet_CV.py -f Kdp Zdr #run CV on elastic net using Kdp and Zdr as features
./convert2xgb.pl data/aggmean_cleaned_smalltrain_manyfeature.csv > data/aggmean_cleaned_smalltrain_manyfeature_xgb.txt
./xgboost_CV.py #run CV on XGBoost using many features
./mixReg_CV.py #run CV on mixture regression
#Geneartion of predictions
./lasso_pred.py #lasso prediction
./elasticNet_pred.py #elastic net prediction
./convert2xgb.pl data/aggmean_cleaned_smalltest_manyfeature.csv > data/aggmean_cleaned_smalltest_manyfeature_xgb.txt #prepare test data
head -n 5000 data/aggmean_cleaned_smalltrain_manyfeature_xgb.txt > data/aggmean_cleaned_smalltrain_manyfeature_xgb_part1.txt #prepare training data
tail -n 5000 data/aggmean_cleaned_smalltrain_manyfeature_xgb.txt > data/aggmean_cleaned_smalltrain_manyfeature_xgb_part2.txt #prepare evaluation data 
./xgboost_pred.py #xgboost prediction
./mixReg_pred.py #mixture regression prediction

#to Perform the following operations on using theses models:
#Linear ridge, Bayesian ridge, random forest regression, gradient boosting regression and ransac
./models2.py
#Run parameter tuning and cross validation using the above models.
./create_features2.py
#Train model using the above models with train data, then perform prediction on test data; prediction result will be used as new features, together with existing features, for ensemble model.
