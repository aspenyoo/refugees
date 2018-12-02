# Spatial and Temporal Predictors of Asylum Outcomes
### Emily Boeke, Yi-Hsuan Fu, and Aspen Yoo 
### Advisors: Daniel Chen, Elliott Ash

### Introduction

The United States government has the power to grant asylum to refugees, foreigners who are in danger of persecution in their home countries. Previous research has indicated that asylum grant rates (in general and per nationality) vary substantially by judge, courthouse, and region of the country where the decision is made (Ramji-Nogales, Schoenholtz, and Schrag, 2008). This regional and inter-judge variability in grant rate is unsettling, as it implies that those seeking asylum may not be receiving equal treatment under the law—in other words, similar cases are not being treated similarly by different judges or in different regions of the country (Ramji-Nogales et al., 2008).

Machine learning methods are of interest in considering this issue for two reasons—(1) these methods allow us to make predictions about the outcomes of asylum cases, which is valuable given the importance of the decision in the applicant’s life, and (2) an examination of the contribution of features to the algorithm’s decision can give some insight into what factors may be shaping judges’ decisions.

In this project, we have investigated the degree to which asylum decisions can be predicted and what features influence judges’ decisions, with special attention to temporal and spatial components of the data. We used US immigration court records from 1985 to 2013, including scheduling, proceeding, and application information pertaining to asylum decisions. 

We begin by testing a model with no spatial or temporal features, using only characteristics of the case and judge identifier. We next test the effect of adding in a number of space- and time-related features to see how these features influence model performance. Others (Chen and Eagel, 2017; Dunn, Sagun, Sirin, and Chen, 2017) have applied machine learning methods to predict asylum case outcomes, but have not specifically focused on the role of temporal and spatial features. Another question explored in our project is the notion of early vs late predictability. That is, how accurately can outcomes be predicted using features available only at the time of the first hearing? How accurately can outcomes be predicted using features available going into the last hearing? 


### Data and Data Prep

Immigration court data from 1985-2013 were obtained from the government via a Freedom of Information Act (FOIA) request issued by the Transactional Records Access Clearinghouse. We used a data table with a record for each immigration court application (Application Table, 4,559,071 records), a table with a record for each scheduled hearing in immigration court (Schedule Table, 6,725,795 records), and a table with a record for each immigration court proceeding (Master Table, 6,084,423 records). The final cleaned and merged dataset consisted of 548,371 records, with each record pertaining to an asylum-related proceeding.

A single “wrapper” [script](https://github.com/aspenyoo/refugees/blob/master/Asylum_analysis_wrapper.py) in python that runs all parts of the analysis . We used the Application Table to determine whether or not a given proceeding was an asylum proceeding, and merged this table with the Master Table. Cleaning and merging these tables can be found in this [script](https://github.com/aspenyoo/refugees/blob/master/Cleaning_asylum.py). From the schedule table, we generated these features: scheduled hearing duration (minutes), average scheduled hearing duration (minutes), case duration (completion date - notice to appear (NTA) date in days), number of days elapsed between first and last hearing, hearing day of week, month of hearing, year of hearing, total number of hearings per proceeding, political affiliation of the president on the date of the hearing. The features were selected from a different hearing depending on the model. For “early'' models, we took information from the schedule corresponding to the first hearing as features for the model. For “late'' models, we took information from the schedule corresponding to the last hearing as features for the model. The Schedule Table cleaning and merging can be found in [these](https://github.com/aspenyoo/refugees/blob/master/Cleaning_detailed_schedule.py) [scripts](https://github.com/aspenyoo/refugees/blob/master/Cleaning_schedule.py).

Given our goal of capitalizing on temporal patterns in judge decisions, we created several moving average features that captured how these decisions evolved over time. We were interested in whether recent history in grant rate for a given judge, nationality, or city would be predictive of the current decision. Here we differentiate between early predictability and late predictability. For the early predictability models, the recent grant rate features are calculated based on the Notice to Appear date of the case. For the late predictability models, the recent grant rate features are calculated based on the completion date of the case (as those can be separated by months or even years). For nationality and city, we created features indexing grant rate in the past year for that feature. Additionally, we made a feature indexing the total number of decisions (grant or deny) for that category type (nationality, city) in the last year. We created dummy variables indicating missing data. For judge, we made features indexing grant rate both in the last year and in the last 10 decisions. These feature generation steps are carried out in this [script](https://github.com/aspenyoo/refugees/blob/master/time_features.py).

The full set of features (separated by early, late, spatial, temporal) is listed in the table below. 

|   |   |  |  |
|------------ | -------------| -------------| -------------|
|**Baseline**<br>All models<br><br><br><br> | **Spatial**<br>Early-S<br>Early-ST<br>Late-S<br>Late-ST| **Temporal**<br>Early-T<br>Early-ST<br>Late-T<br>Late-ST | **Late**<br>Late-S<br>Late-T<br>Late-ST<br><br>|
|Judge ID<br>Nationality<br>Attorney present<br>Asylum case type<br>Case type<br>Type of hearing<br>Adj medium of hearing<br>Language of hearing<br><br><br><br><br><br><br><br><br><br><br><br><br><br>| Court ID<br>Base city<br>Hearing city<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>| NTA date<br>President’s political affiliation<br>during hearing<br>Hearing year<br>Hearing month<br>Hearing day of week<br>Scheduled hearing duration<br>Grant year in the last<br>year for given nationality<br>Number of decisions in the<br>last year for a given<br>nationality<br>Grant rate in the last year<br>for a given judge<br>Number of decisions in the<br>last year for a given judge<br>Grant rate in the last<br>year for a given base city<br>Number of decisions in the<br>last year for a given<br>base city| Num hearings per proceeding<br>Days between first and last<br>hearing of proceeding <br>Days between case completion<br>and NTA date <br>Grant rate for last 10<br>decisions for a given judge<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>|

**Table 1.** Features, sorted by model type. Under each class of features, we list the models that contain these features (see Model Selection section for guide to codes). Notes: Hearing year, month, day of week, and president all correspond to either the first or last hearing, depending on whether early or late predictability analysis).  All grant rate features have dummy variables indicating whether there is missing data in that time frame.

### Model Selection
Model comparison was carried out in [these](https://github.com/aspenyoo/refugees/blob/master/Log_Reg_Models.py#L11) [scripts](https://github.com/aspenyoo/refugees/blob/master/Full_Model_H2O.py). The outcome variable is whether an applicant was granted or denied asylum. Thus, this is a classification problem. We created 14 models by factorially combining three model attributes. First, we used one of two classification algorithms to predict the outcome of the asylum case: logistic regression (via scikit-learn) or random forest (via H2O). Second, we tested a model with neither spatial nor temporal features (baseline), adding only spatial features to the baseline features, adding only temporal features to the baseline features, and adding both spatial and temporal features to baseline. We did this to investigate whether temporal or spatial features helped predict the outcome above and beyond other features considered. Third, we were interested in how the model performed using features only available early versus late in the unfolding of the case. Early models used only information that was available before the first hearing; late models used information available before the last hearing. This factorial combination leads to 2 (model: logistic regression, random forest) x 4 (features: baseline, spatial, temporal, spatial temporal) x 2 (predictability: early, late) = 16 models. However, the baseline models for early and late predictability are identical, so we have 14 total models. For clarity, we refer to the models by their predictability (Early vs. Late), whether spatial/temporal features were added (B: baseline, S: spatial, T: temporal, ST: spatial and temporal), and the type of algorithm (LR: logistic regression, RF: random forest). For example, the model “Late-S-LR'' is a late predictability logistic regression with spatial features added. The features included in this model are in table 1. 

There are a few potential limitations of working with regression models on this dataset. First, logistic regressions do not work well with categorical data with many values. We one-hot encoded all categorical features in order to fit this model, which resulted in over 1300 features. Another potential limitation is that logistic regressions assume a linear → sigmoid relationship between variables and grant rate, which may not be the case. The Random Forest Classifiers do not suffer from these limitations. They handle categorical features very well--a node can be split by category, without one-hot encoding. Because Random Forest classifiers don’t necessitate one hot encoding, they make model interpretation simpler and cut down on the number of features that must be kept in memory. They also naturally capture interactions between features--interactions don’t have to be explicitly coded, as they do in logistic regression. 
    
*Model evaluation, hyperparameter tuning, and comparison*

We began by splitting the data into a training set (80% of sample) and a test set. All model comparison and hyperparameter tuning was done using 5-fold cross validation in the training dataset. Since our classes are imbalanced (40% grant, 60% deny), we evaluated models with Area Under the Curve (AUC). We trained the model and completed hyperparameter search using five-fold cross validation. We conducted a hyperparameter grid search over regularization penalty (L1, L2) and inverse regularization parameter (10^-10 and 10^2). For random forest models, we tuned number of estimators (20, 50, 100, or 200). 

|  |Early Predictability	| Late Predictability|
|------------ | -------------| ------------ |
|Baseline (B-LR) |	77.0%|
|Spatial (S-LR)	|76.6%	|77.5%|
|Temporal (T-LR)	|85.2%	|85.6%|
|Spatial + Temporal (ST-LR)|	84.3%	|85.5%|

**Table 2.** Mean of five-fold cross validation scores for logistic regression models.

|  |Early Predictability	| Late Predictability|
|------------ | -------------| ------------ |
|Baseline (B-RF) |	87.6%|
|Spatial (S-RF)	|88.0%	|88.8%|
|Temporal (T-RF)	|90.3%	|91.6%|
|Spatial + Temporal (ST-RF)|90.3%	|91.6%|

**Table 3.** Mean of five-fold cross validation scores for random forest models. 

We found that random forest models substantially outperformed the logistic regression models. However, within each model class, the patterns of how different features influenced performance were similar. Late models outperformed early models in all cases, but by a very small margin. Adding temporal features led to a substantial increase in accuracy. Adding spatial features had little effect: The models with spatial features performed comparably to the baseline models, and spatial + temporal models performed similarly to temporal models.

We planned to test the performance of the winning early and late models using the hold out dataset (20% of the data). Given how many models we tested, we wanted to ensure that final test performance was an unbiased estimate of how well the model would generalize. We assumed a clear winning model would emerge from our cross validation results, but the T-RF and ST-RF models performed nearly identically (differences ~0.001). Since the models were tied, we picked the more parsimonious model (T-RF) to evaluate in the test set. The random forest model performed with an AUC of 90.8% for early predictability, and 92.1% for late.

We investigated the variable importances of the early and late T-RF models, which are plotted in Figures 1 and 2. In both cases, judge identifier and nationality were the highest-ranking features. Our hand engineered features that tracked grant history in the past year for a given nationality, judge, or city also played a prominent role in the model. Other temporal features, like hearing year, and NTA date also had high importance. One non-temporal feature that had a high importance was hearing language. The feature importance scores for the early and late model were similar.

![Figure 1](https://github.com/aspenyoo/refugees/blob/master/fig1.png)

**Figure 1.** Variable importance for the Early Temporal Random Forest model (Early-T-RF). tracid = judge identifier. nat = nationality, nat_gr_last1yr_early = grant rate for given nationality in last year.

![Figure 2](https://github.com/aspenyoo/refugees/blob/master/fig2.png)

**Figure 2.** Variable importance for the Late Temporal Random Forest model (Late-T-RF). 

### Discussion
We investigated what features could predict whether a refugee was granted asylum in the United States. In particular, we wanted to see whether spatial and temporal aspects of a case could predict its outcome. We were able to predict asylum outcomes with high accuracy (AUC 92.1%).

Temporal features contributed substantially to model predictions. For example, the AUC for early logistic regression jumped from 77.7% to 85.2% by adding temporal features. The results suggest that grant decisions fluctuate substantially over time. Features that tracked grant rate for a given nationality, city, or judge had very high importance scores. This suggests that judges, nationalities, and cities have idiosyncratic trajectories in their grant rates across time. In the case of nationalities, this is rather logical: if a country has a humanitarian crisis, the next few years may yield an uptick in grant rate for that nationality.

Spatial features, on the other hand, did not seem to contribute to the model predictions. The lack of contribution of spatial features may reflect that the information contained in the spatial features (city, courthouse) is already contained in the judge identifier feature. Also, were only 3 spatial features, and we devoted more energy to developing temporal features, so it is possible that spatial features could contribute more to predictions with further development and feature generation. 

There were also some non-spatial or temporal features that played prominent roles in the model. Judge identifier and nationality were consistently the most important features. This large discrepancy between judges was highlighted by Ramji-Nogales et al (2008). 

Surprisingly, we did not find that late models tended to perform much better than early models, suggesting not only that the outcome of a case can be predicted before it has begun, but that cases are not much more predictable with the information available (at least with the data we had) right before the case closes. This is consistent with the findings of Dunn et al (2017). The lack of an accuracy gap between early and late models could be explained by the many unobserved variables that shape judge’s decisions (e.g., whether the applicant met the criteria for meriting asylum). However, it remains unsettling that judges’ decisions can be predicted so well from information available before the merits of the case have been heard. 


### References
Chen, DL and Eagel, J. (2017). Can Machine Learning Help Predict the Outcome of Asylum Adjudications? *Proceedings of the 16th edition of the International Conference on Artificial Intelligence and Law*: 237-240.

Dunn, M, Sagun, L, Sirin, H, and Chen, D. (2017). Early Predictability of Asylum Court Decisions. *Proceedings of the 16th edition of the International Conference on Artificial Intelligence and Law*: 233-236.

Ramji-Nogales, J, Schoenholtz, A, and Schrag PG (2008). Refugee Roulette: Disparities in Asylum Adjudication. *Stanford Law Review*, 60(2):295-412.


