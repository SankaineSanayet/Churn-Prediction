# Churn-Prediction
This project is part of the Moringa Phase 3 final project
# Overview
Customer churn is a critical challenge for businesses, especially in the telecommunications industry. This project focuses on predicting customer churn for SyriaTel, a telecom company, using machine learning techniques. The goal is to build a model that
can accurately classify whether a customer is likely to churn, allowing the company to take proactive measures to retain them.

By identifying the most influential factors contributing to customer churn, this project provides valuable insights that can
help SyriaTel develop targeted retention strategies.

# Business Understanding
The objective of this project is to predict customer churn for SyriaTel, a telecommunications company. The goal of this analysis is to be able to build a model that accurately predicts the customers that are likely to churn so that SyriaTel can take proactive measures to retain them leading to long term revenue generation and company growth.

# Data Understanding
The dataset we are working with for this project has been obtained from [Kaggle](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset). The dataset contains 3333 records and 21 columns. Out of the 17 columns, 4 are categorical features and 17 numerical features.

# Data Preprocessing
In this phase, we prepared the dataset for modeling by encoding categorical variables, scaling features, removing outliers and performing feature engineering.
The final step was splitting the data into training and testing set in preparation for modeling

# Modeling
The Logistic Regression has been used as our base model for performance comparison purposes with the rest of the models.
We then went ahead to build Decision Tree model, Random Forest Classifier and XG Boost so as to evaluate their performances in predicting churn rate for Syria Tel Company.

# Model Evaluation
The results table shows that the LogisticRegression has the highest recall score, followed by DecisionTreeClassifier then RandomForestClassifier and XGBClassiefier.

The ROC curve analysis shows that the RandomForestClassifier and XGBClassifier have the best performance with auc score of 0.91, followed by the  DecisionTreeClassifier and LogisticRegression at 0.83. 

# Model Tuning
In this phase, we tune the Decision Tree and Random Forest Classifier models in an attempt to improve their performance at predicting churn rate at Syria Tel.

After conducting hyperparameter tuning using GridSearchCV, we evaluated the models based on the recall score

Recall score for the tuned Random Forest model improves to 0.71 with an ROC score of 0.91, an improvement from the untuned model. This means that the R.F model is better at correctly identifying positive instances (true positives) while
maintaining a good balance between correctly classifying both positive and negative instances. 
The ROC AUC score indicates the model's strong performance in distinguishing between the two classes across different decision threshold.

The recall score for the tuned Decision Tree model slightly drops to 0.69 with an ROC score of 0.88. which is an improvement from the untuned model.
Model misses a few more positive instances compared to the untuned model. However, the improvement in the ROC AUC score (0.88) shows that the model has improved its ability to distinguish between positive and negative instances overall,
indicating a better overall classification performance.

# Conclusion
Random Forest is the superior model in this case, with a better recall and ROC AUC score, making it more reliable for identifying customers who are likely to churn
Both models benefit from tuning, with improvements in their ability to distinguish between positive and negative instances
Total Charges, Customer Service Calls, and International Plan are important features influencing customer churn predictions

#Recommendation
 - Targeted Promotions and Discounts: Use the model to identify customer segments at risk of churning and offer them targeted discounts, loyalty programs, or bundled packages that address their specific needs and usage patterns. For example customers from area 415
 - Improved Plan Pricing Strategies: The company should consider providing better prices for international and voicemail plans as most customers who churned had not subscribed to either of the plans 
 - Significant Features Enhancement: SyriaTel should focus on enhancing the features identified as significant in predicting the churn so as to increase customer retention ie call charges 
 - Customer Retention Strategies: The company should focus retention strategies in states where there was a high churn rate

# For More Information:
Review the full analysis in [Jupyter Notebook](https://github.com/SankaineSanayet/Churn-Prediction/blob/main/index-checkpoint.ipynb) or the Presentation
