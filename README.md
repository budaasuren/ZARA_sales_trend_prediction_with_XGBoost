# ZARA_sales_trend_prediction_with_XGBoost

In this project, I have completed the feature engineering and Machine Learning Model Training with XGBoost in a real world complex data. 

The main goal is to predict best performing products for the last week in terms of revenue using previous weeks data.

The projects contains EDA, Feature Engineering and Model Training parts.

EDA
1. EDA shows that any given producs have multiple positions on the website.
![image](https://user-images.githubusercontent.com/113545468/233159019-8c396ef9-53c0-47e2-a06b-eae32ea5c630.png)
2. Similarly, for the block column, EDA shows multiple products belong to a certain block. 
![image](https://user-images.githubusercontent.com/113545468/233159183-742e31b8-6268-4e42-ba52-afed9da04700.png)

Feature Engineering 
1. EDA shows that any given producs have multiple positions on the website. So I decided to aggregate the position . Added the several aggregation results into the original data as additional features.

![Screen Shot 2023-04-19 at 1 54 27 PM](https://user-images.githubusercontent.com/113545468/233159665-8bde61a4-0598-4f2c-b5a9-7f07fa74e059.png)

2. Similarly, for the block column, EDA shows multiple products belong to a certain block. So again I decided to aggregate the data and aded new columns as num_items (new feature).

![Screen Shot 2023-04-19 at 1 55 00 PM](https://user-images.githubusercontent.com/113545468/233159786-e967ebcf-d1ab-444f-8689-ec3a87a97a40.png)

3. As the scope of the project in product level, I removed the color_id and size_id columns. Then aggregated the sales and stock columns by product only.

![Screen Shot 2023-04-19 at 1 56 13 PM](https://user-images.githubusercontent.com/113545468/233160071-52ee053a-1c24-48b1-99e4-7969b90016e0.png)


4. As the last feature engineering, I chose the last day in the training data ( day 84). Then used unsupervised ML ( to be specific k-means (to find the optimal number of clusters) and PCA (to reduce the dimension and as a result it will give us more clean clusters)) . Plotted the results and added the cluster column to the original data as an another new feature. Interestingly, this new feature ended up being second important feature in the last (or best) model.

![Screen Shot 2023-04-19 at 1 56 50 PM](https://user-images.githubusercontent.com/113545468/233160215-f5f3e2b4-da4b-4d84-a088-23e54827f6f2.png)


5.Finally, I merged all the transformed columns and created final dataframe for the model training.

![Screen Shot 2023-04-19 at 1 57 22 PM](https://user-images.githubusercontent.com/113545468/233160356-8b8ffdbd-29b6-4a2f-b4d4-a4743baa858a.png)


Model training:

* I used 3 different ML models  which are RF, XG Boost and DL. 
* Classification models such as Logistic Regression would not work in this case since the target variable is numerical. 
* Also I did not experiment with linear regression because this will go very wrong as we were not able to normalize the categorical variables using dummy variables (0 ,1). The reason is that the dataset has over 15000 products to convert. Even I convert them, linear regression will still suffer from dimensional issue. 
* For this project, tree based models work better as they do not depend on data normalization.  These models consider each regressor independently unlike linear regression which regressors will have interdependency with each other where data normalization is necessary.

![image](https://user-images.githubusercontent.com/113545468/233161088-d9a38e5b-c6da-4f72-ae1d-5f629e31dab9.png)


![Screen Shot 2023-04-19 at 2 00 19 PM](https://user-images.githubusercontent.com/113545468/233161040-559e5b8e-8b09-43a8-b416-5e93ae6c159b.png)

* Experimented with RF and the r score was 0.4.
* Experimented with XG Boost and R score was 0.5.(Model training with XGBoost (experiment1: n_estimators=100, max_depth=7)
* Model training with XGBoost (experiment2: n_estimators=200, max_depth=10) and the r score increased to 0.59.
*  Model training with XGBoost (experiment 3: n_estimators=300, max_depth=10) and the r score increased to 0.60.
* Model building with XG boost with fewer features ( removed the least important features hoping it will give higher r score) . The r score increased to 0.61 . I stopped the experiments here with XG Boost as I can clearly see the results has converged already and it is obvious that it will not improve much with further optimizations. This will the best model.
* Lastly, I experimented wit Deep Learning and the result was really bad as expected. R score : -3.340532347850811e-06. I only used 10 epoch because I already observed the conversion in the result and the running time would be hours. DL generally does not do well in structured data (works better for images or natural language procession).


Prediction:
Created the function that I reused to calculate the overlapping percentage of predicted vs actual 100 products that produced the most amount of revenue. To do that added the revenue columns for both train and test data.
One observation from the prediction that was interesting to me was that there was a tendency for prediction accuracy for top 100 performing products to decrease over time. It make sense because we can more accurately predict near future than far future.
Conclusion:
For the EDA part, day 84 cluster’s output label was the second important feature after stock (most important feature).
Among the models we experimented, XGBoost provided the best performing model and Deep Learning model was the worst performing model. Moreover, the Random Forest model was the medium one. Classification models such as Logistic Regression would not work in this case since our target variable is numerical. Linear regression would not handle the dimensionality of the explanatory variables in this data.
Given the data ( real world complex data ), I believe the r-squared score of 0.61 is good enough number. That basically indicates that our model explains 61% of variation (how the target variable moves with the changes in explanatory variables) in target variable of sales in test data. In other words, we are talking about the model fitness here. In real industry economic application, r-squared value of 0.61 is considered a reasonably good fit.
About Top 100 best performing products, we predicted 50-60% accurately. And the further forecasting horizon, the lower the accuracy.
