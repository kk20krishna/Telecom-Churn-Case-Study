# Telecom-Churn-Case-Study

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


