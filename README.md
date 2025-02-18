# Telecom Churn Case Study

# Problem statement

In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.

For many incumbent operators, retaining high profitable customers is the number one business
goal. To reduce customer churn, telecom companies need to predict which customers are at high risk of churn. In this project, you will analyze customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn, and identify the main indicators of churn.

In this case study, our goal is *to build a machine learning model that is able to predict churning customers based on the features provided for their usage.*

**Objectives**
The main goal of the case study is to build ML models to predict churn. The predictive model that you’re going to build will the following purposes:

1. It will be used to predict whether a high-value customer will churn or not, in near future (i.e. churn phase). By knowing this, the company can take action steps such as providing special plans, discounts on recharge etc.

2. It will be used to identify important variables that are strong predictors of churn. These variables may also indicate why customers choose to switch to other networks.

3. Recommend strategies to manage customer churn based on your observations.

**Customer behaviour during churn:**

Customers usually do not decide to switch to another competitor instantly, but rather over a
period of time (this is especially applicable to high-value customers). In churn prediction, we
assume that there are three phases of customer lifecycle :

1. <u>The ‘good’ phase:</u> In this phase, the customer is happy with the service and behaves as usual.

2. <u>The ‘action’ phase:</u> The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. It is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)

3. <u>The ‘churn’ phase:</u> In this phase, the customer is said to have churned. In this case, since you are working over a four-month window, the first two months are the ‘good’ phase, the third month is the ‘action’ phase, while the fourth month (September) is the ‘churn’ phase.

# Column Transformer and Pipeline Approach for Robust Data Preprocessing

***This notebook showcases a robust and production-ready approach to data preprocessing using the ColumnTransformer and Pipeline classes from scikit-learn. Unlike approaches that often involve directly modifying the test data, this approach focuses on defining a sequence of preprocessing and feature engineering steps that are then applied within a pipeline structure. This modularity and encapsulation offer several benefits, making the process more efficient and reliable.***

![alt text](https://github.com/kk20krishna/Telecom-Churn-Case-Study/blob/main/Pipeline.png?raw=true)

**Distinct Advantages of Pipeline Approach**

1. Data Integrity and Reproducibility

2. Modularity and Flexibility

3. Production Readiness



**Application in this Notebook**

In this notebook, the preprocessing and feature engineering steps are first clearly defined using separate functions or custom transformer classes. These steps are then encapsulated within a Pipeline using the ColumnTransformer. This pipeline handles various tasks, including:
- Custom Feature Engineering on Date Columns: Feature engineering is applied to specific date columns, calculating differences and creating new features, which are then imputed if required.
- Imputation: SimpleImputer handles missing values in both custom and generic imputation columns.
- Dropping Columns: Unnecessary columns are dropped using drop.
- Passthrough for Unchanged Columns: Unchanged columns are passed through as is using passthrough.

The resulting pipeline ensures that raw input data undergoes the same set of transformations before being fed into the models, making the process robust and consistent.

**Overall Approach**

This approach differs from the approach of modifying test data, directly by separating preprocessing steps from model training  allow these steps to be captured within a reusable pipeline. This makes the code cleaner, more maintainable, and less prone to errors. It also ensures that the same transformations are applied consistently across different datasets.

With its modularity, flexibility, and production readiness, this approach is ideal for building reliable and deployable machine learning models, making it particularly well-suited for deployment in cloud environments like Azure or AWS. It allows for effortless transition from experimentation to production, ensuring the consistent and predictable performance of models over time.

# Models and Metrics
Based on the objectives of the case study, we will be creating two sets of models.
**For objectives #2 and #3 we will create:**

> Objective# 2: It will be used to identify important variables that are strong predictors of churn. These variables may also indicate why customers choose to switch to other networks.

> Objective# 3 Recommend strategies to manage customer churn based on your observations.

**Interpretative Models:** These models aim to provide insights into the factors that contribute to churn.
  - **F1 score** will be the evaluation metric for interpretative models because it balances precision and recall, ensuring that the insights derived from the model are both reliable and informative.

**For objective #1 we will create:**
> Objective# 1 It will be used to predict whether a high-value customer will churn or not, in near future (i.e. churn phase). By knowing this, the company can take action steps such as providing special plans, discounts on recharge etc.

**Predictive Models:** These models will predict the likelihood of churn for individual customers.

  - **Accuracy** will be the evaluation metric for predictive models.


***In summary, while F1 score ensures the model is robust and balanced in detecting churn events in interpretative models, accuracy is ideal for predictive models where the goal is to reliably forecast customer churn across both churners and non-churners.***

## Feature importances and Recomendations
The **Logistic Regression model** provides valuable insights into the key factors influencing customer churn. By examining the feature coefficients, we can identify the variables most strongly associated with churn and understand their impact. These insights can guide customer retention strategies.

**Key Phases of Customer Lifecycle:**
1. **Good Phase:** Customers are satisfied and behave as usual.
2. **Action Phase:** Customers exhibit early warning signs of dissatisfaction or interest in competitors. This phase is critical for intervention.
3. **Churn Phase:** Customers decide to leave the network.

For the dataset, months 6 and 7 represent the good phase, month 8 represents the action phase, and month 9 (September) is the churn phase.

***Let us focus on the action phase and determine the influencing factors. This will enable appropriate action to be taken in the action phase.***
### **Top 10 Positive Contributors (Increase Likelihood of Churn)**

These features from the action phase (Month 8) have the largest positive coefficients, indicating they increase the probability of churn:

- **ARPU in Action Phase (Month 8):** Low average revenue per user in the action phase suggests reduced customer engagement.
- **Outgoing On-Net Calls (MOU) in Action Phase (Month 8):** A decline in on-network call usage during the action phase reflects lower dependence on the service.
- **Outgoing ISD Calls (MOU) in Action Phase (Month 8):** Fewer international calls during the action phase indicate reduced usage.
- **Incoming Off-Net Calls (MOU) in Action Phase (Month 8):** Reduced off-network incoming calls during the action phase highlight declining interaction.
- **Total Outgoing Call Usage (MOU) in Action Phase (Month 8):** Lower outgoing call activity in the action phase is a significant churn predictor.
- **Outgoing Local Calls to Mobile Numbers (MOU) in Action Phase (Month 8):** - Declining local call activity signals waning reliance on the network.
- **Last Recharge Amount in Action Phase (Month 8):** Lower recharge amounts during the action phase reflect disengagement.
- **Incoming On-Net Calls (MOU) in Action Phase (Month 8):** Fewer on-network incoming calls point to reduced customer interaction.
- **Incoming Off-Net Minutes of Use (MOU) in Action Phase (Month 8):** Lower off-network call minutes during the action phase suggest reduced engagement.
- **Total Recharge Amount in Action Phase (Month 8):** Decreased recharge amounts during the action phase indicate reduced likelihood of continued usage.
### **Top 10 Negative Contributors (Decrease Likelihood of Churn)**

These features from the action phase (Month 8) have the largest negative coefficients, meaning they decrease the probability of churn:

- **Trend in Total Recharge Amount**: An increasing recharge trend during the action phase signals continued customer satisfaction.
Outgoing Local Calls to Mobile Numbers (MOU) in Action Phase (Month 8): High local call usage during the action phase reduces churn risk.
- **Total Outgoing Call Usage (MOU) in Action Phase (Month 8):** Higher outgoing call activity in the action phase reflects active engagement.
- **Outgoing Local Calls (MOU) in Action Phase (Month 8):** High local outgoing calls during the action phase indicate sustained usage.
- **Local Incoming Calls (MOU) in Action Phase (Month 8):** Higher local incoming call usage signals ongoing interaction with the network.
- **Trend in Outgoing ISD Calls (MOU):** A positive trend in international calls during the action phase lowers churn likelihood.
- **Total Data Usage (MB) in Action Phase (Month 8):** Active data usage in the action phase suggests strong engagement.
- **Outgoing On-Net Calls (MOU) in Action Phase (Month 8):** Increased on-network outgoing call activity reflects customer reliance on the network.
- **Last Recharge Amount in Action Phase (Month 8):** Large recharge amounts during the action phase indicate satisfaction with the service.
- **Incoming On-Net Calls (MOU) in Action Phase (Month 8):** High on-network incoming calls during the action phase point to strong customer interactions.
### **Insights on Customer Behavior**
- **Behavioral Shifts:** The strongest churn predictors often relate to a noticeable decline in recharge activity, ARPU, and call/data usage during the action phase (month 8). These shifts signal customer dissatisfaction or reduced dependency on the network.
- **Engagement:** Features like high call activity and consistent recharge patterns during the action phase suggest continued customer engagement, lowering churn risk.
- **Data Usage:** Reduced frequency in data recharges and usage in the action phase is a key indicator of churn, reflecting the increasing reliance on data services in customer behavior.
### **Recomendations**

1. **Proactive Interventions:**
Focus on customers exhibiting early warning signs in the action phase (e.g., declining ARPU, reduced recharge activity).
Offer personalized retention strategies such as discounts, data/call packages, or loyalty programs.

2. **Targeted Campaigns:**
Design campaigns to re-engage customers with low ARPU and recharge frequency during the action phase.
Target high-value customers who show early signs of churn for priority interventions.

3. **Customer Monitoring:**
Continuously monitor key metrics like ARPU, recharge patterns, and call/data usage trends.
Automate alerts for customers showing significant deviations from their historical behavior.

# Accuracy
***We are selecting the hyperparameter tuned XGB model for the Kaggle submission since this is the model giving us the highest accuracy.***
https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fcompetitions%2Ftelecom-churn-case-study-hackathon-c-67%2Fleaderboard
