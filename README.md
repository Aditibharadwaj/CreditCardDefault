# Credit Card Score Modeling via Classification & Risk Techniques
Project Approach & Modeling Strategy
This project follows a complete, end-to-end workflow for building a credit card default prediction model. The approach includes loading and understanding the data, performing EDA, handling imbalance, creating financial-behavior features, training multiple models, tuning thresholds, and selecting the best-performing classifier.

1. Data Loading & Overview

   Imported the dataset and inspected structure, missing values, and data types.
   Dataset contains 26 features (excluding customer ID).
   Age column had 126 missing values → filled using median due to skew.
   Duplicate records removed.
   Inconsistent categories found in Education (7 values) and Marriage (4 values) → grouped and corrected.
   Default Distribution
   Total customers: 25,247
   Defaults: 4,807 (19%), Non-defaults: 20,440 (81%)
   Strong class imbalance → addressed using SMOTE, class weights, and threshold tuning.

2. Exploratory Data Analysis (Key Findings)
   
   Credit Limit (LIMIT_BAL)
   Most customers have limits below ₹2 lakh; distribution is right-skewed.
   Defaulters generally have lower credit limits.
   Higher credit limits (> ₹4 lakh) → very low default rates.
   
   Gender
   Females: ~15.2k, Males: ~10k
   Default rates → Males: 21%, Females: 18%
   Females show slightly better repayment patterns.
   
   Education
   Most customers are from University or Graduate School.
   Default rate decreases with higher education.
   High School customers show highest default rate (~21%).
   
   Age
   Majority between 20–40 years.
   Younger groups show more defaults → possibly due to unstable income.
   Payment Status (PAY_0 to PAY_6)
   Most payments are on time (0), or early (-1, -2).
   Late payments strongly correlate with default → important predictive feature.
   Bill Amount Patterns
   Bill amounts are right-skewed; highly correlated across months.
   Defaulters exist across all bill ranges → bill amount alone is not a good separator.
   
   Outliers
   Extreme values detected in bill & payment columns → clipped to reduce noise.
   
   Correlation Heatmap
   PAY_0 has strongest correlation with default (r ≈ 0.32).
   Past payment delays also correlate moderately.
   Bill amounts highly correlated with each other (r > 0.85).
   Demographic variables show weak predictive power.

3. Data Cleaning & Preparation

   Filled missing ages using median.
   Cleaned inconsistent categories for Education and Marriage.
   Removed duplicates.
   Applied SMOTE to fix imbalance in training data.

4. Feature Engineering
   
   A. Reducing Multicollinearity
   Removed highly correlated features (corr > 0.7).
   Kept only selected billing and payment status features.
   Result → simpler, more stable, less noisy model.
   
   B. Engineered Behavioral Features
   Created financial behavior–oriented features such as:
   CV_bill (variability in monthly bills)
   CV_pay (payment variability)
   Frequency of zero payments
   Average & Max payment delay
   Credit Utilization Ratio
   Total Bill – Total Payment
   Repayment consistency score
   These features improve model’s ability to detect risky customers.

5. Model Training
   
   Trained and compared:
   XGBoost
   Decision Trees
   Logistic Regression
   
   Evaluation metrics:
   F2 Score
   AUC-ROC
   Accuracy 

6. Threshold Tuning
   Used precision–recall curve to evaluate thresholds.
   Optimized for F2-score (recall-weighted) because missing defaulters is costly.
   Best threshold for XGBoost: 0.198
   Result → significantly higher recall with balanced precision.

   <img width="951" height="727" alt="image" src="https://github.com/user-attachments/assets/c29962d5-a767-4ca3-b17d-1437116d93f4" />


   Final Model: XGBoost
   F2-score: 0.89
   AUC-ROC: 0.946
   High recall (94%), good precision (76%)
   Best suited for real-world credit risk analysis.

   
