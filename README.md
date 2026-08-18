Credit Card Default Prediction using Decision Tree

A machine-learning classification project focused on predicting whether a credit-card customer will default on their payment in the following month.

The project goes beyond simply training a Decision Tree. It investigates overfitting, tree complexity, hyperparameter tuning, cross-validation, class imbalance, probability thresholds, feature engineering, feature importance, and business-oriented metric selection.

Project Objective

Predict:

0 → No default
1 → Default

The initial objective was general classification performance.

During model evaluation, however, it became clear that accuracy alone was not sufficient because the dataset contained substantially more non-defaulters than defaulters.

The project therefore shifted toward:

Improving the model's ability to identify actual defaulters.

The primary metric became recall for class 1.

Dataset

The dataset contains:

30,000 rows
26 predictor features

After splitting:

Training: 24,000
Testing:   6,000

The split used:

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
Workflow
Data Understanding
       ↓
EDA
       ↓
Correlation Analysis
       ↓
Feature Engineering
       ↓
Train/Test Split
       ↓
Baseline Decision Tree
       ↓
Overfitting Investigation
       ↓
Hyperparameter Experiments
       ↓
Cross-Validation
       ↓
Class Imbalance
       ↓
Recall Optimization
       ↓
Threshold Analysis
       ↓
Feature Importance
       ↓
Final Evaluation
1. Baseline Decision Tree

The initial Decision Tree produced:

Training Accuracy ≈ 99.95%
Testing Accuracy  ≈ 72.17%

This large gap indicated significant overfitting.

Tree complexity was:

Depth: 52
Leaves: 3764

The model was therefore learning extremely specific patterns from the training data.

2. Controlling Tree Complexity

Three major Decision Tree controls were investigated.

max_depth

Increasing depth caused training performance to increase while test performance eventually decreased.

Example:

Depth  2 → Test 81.68%
Depth  5 → Test 81.88%
Depth 10 → Test 80.82%
Depth 20 → Test 74.40%
Depth 50 → Test 72.30%

This demonstrated the relationship between model complexity and overfitting.

min_samples_split

Increasing the minimum number of samples required for a split reduced overfitting.

2     → Test 72.17%
100   → Test 79.23%
500   → Test 81.12%
1000  → Test 81.57%
2000  → Test 81.62%
min_samples_leaf

Increasing the minimum number of samples allowed in a leaf also reduced overfitting.

1    → Test 72.17%
20   → Test 79.00%
50   → Test 80.95%
100  → Test 81.55%
500  → Test 81.72%
3. Cross-Validation

Repeatedly using the same test set for hyperparameter selection can gradually cause the test set to influence model decisions.

To address this, GridSearchCV was introduced.

The search explored:

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [100, 500, 1000, 2000],
    'min_samples_leaf': [50, 100, 200, 500]
}

Five-fold cross-validation was used.

4. Accuracy vs Recall

An accuracy-optimized model achieved:

Accuracy : 81.90%
Precision: 67.54%
Recall   : 34.97%
F1       : 46.08%

The problem:

The model was only identifying approximately 35% of actual defaulters.

Therefore, optimizing accuracy was not aligned with the project's primary objective.

5. Class Weighting

The Decision Tree was changed to:

class_weight='balanced'

and GridSearchCV was optimized using:

scoring='recall'

Best configuration:

max_depth = 10
min_samples_leaf = 100
min_samples_split = 100

Cross-validation recall:

≈ 66.75%

Test performance:

Accuracy : 70.87%
Precision: 40.44%
Recall   : 67.14%
F1       : 50.48%

Confusion matrix:

[[3361 1312]
 [ 436  891]]

This demonstrated a major trade-off:

Higher recall for defaulters came at the cost of precision and overall accuracy.

6. Probability Threshold Analysis

Instead of relying exclusively on the default classification threshold of 0.50, multiple thresholds were investigated.

Threshold	Accuracy	Precision	Recall	F1
0.20	48.20%	28.01%	85.46%	42.19%
0.30	56.25%	31.00%	79.80%	44.66%
0.40	63.57%	34.39%	71.29%	46.40%
0.50	69.88%	39.25%	66.01%	49.23%
0.55	73.73%	43.36%	61.27%	50.78%
0.60	75.97%	46.44%	56.44%	50.95%

The experiment demonstrated the expected precision-recall trade-off.

Lower thresholds catch more defaulters but create more false positives.

Higher thresholds produce fewer positive predictions but improve precision.

No threshold provided a sufficiently meaningful improvement in the project's primary objective to justify replacing the standard operating point.

7. Feature Engineering

The payment history variables showed strong relationships:

PAY_0 ↔ PAY_2 = 0.67
PAY_2 ↔ PAY_3 = 0.77
PAY_3 ↔ PAY_4 = 0.78
PAY_4 ↔ PAY_5 = 0.82
PAY_5 ↔ PAY_6 = 0.82

This motivated the creation of behavioral summary features.

PAY_MAX
credit['PAY_MAX'] = credit[pay_cols].max(axis=1)
PAY_AVG
credit['PAY_AVG'] = credit[pay_cols].mean(axis=1)
PAY_TREND
credit['PAY_TREND'] = credit['PAY_0'] - credit['PAY_6']
DELAY_COUNT
credit['DELAY_COUNT'] = (credit[pay_cols] > 0).sum(axis=1)
8. Behavioral Insights

DELAY_COUNT showed a strong relationship with default rate:

Number of delayed periods	Default rate
0	11.71%
1	29.82%
2	38.76%
3	50.87%
4	57.31%
5	57.38%
6	70.32%

This suggests that repeated payment delays are strongly associated with future default.

9. Feature Importance

The Decision Tree identified the following major contributors:

Feature	Importance
PAY_0	0.6806
PAY_MAX	0.1708
PAY_AVG	0.0474
PAY_AMT2	0.0243
BILL_AMT1	0.0197
PAY_AMT1	0.0123
LIMIT_BAL	0.0087

PAY_0 was by far the dominant feature.

However, feature importance should not automatically be interpreted as proof that adding a feature improves predictive performance.

10. Feature Engineering Evaluation

After adding:

PAY_MAX
PAY_AVG
PAY_TREND
DELAY_COUNT

the model produced:

Accuracy : 71.43%
Precision: 40.90%
Recall   : 65.56%
F1       : 50.38%

Compared with the previous class-weighted model:

Accuracy : 70.87%
Precision: 40.44%
Recall   : 67.14%
F1       : 50.48%

The engineered model slightly improved accuracy and precision but reduced recall and slightly reduced F1.

Therefore:

Feature engineering did not produce a meaningful improvement for the Decision Tree under this configuration.

This was an important finding rather than a failure.

11. Final Model Assessment

The Decision Tree demonstrated approximately:

Accuracy  ≈ 71%
Precision ≈ 40%
Recall    ≈ 65–67%
F1        ≈ 50%

depending on the configuration and feature set.

The recall-focused class-weighted model was more aligned with the project's objective than the accuracy-optimized model.

12. Key Lessons
Overfitting

A Decision Tree can easily memorize training data when allowed to grow excessively.

Regularization

max_depth, min_samples_split, and min_samples_leaf can restrict tree complexity.

Class imbalance

Accuracy can hide poor performance on the minority class.

Recall

For this problem, recall answers an important question:

Of the customers who actually defaulted, how many did the model identify?

Thresholds

Probability thresholds allow the classification behavior to be adjusted according to the desired precision-recall trade-off.

Feature engineering

Creating additional features does not guarantee better predictive performance.

Cross-validation

Hyperparameters should be selected using training data and cross-validation rather than repeatedly optimizing against the final test set.

13. Limitations

This project has several limitations.

The Decision Tree is a relatively simple model for a complex financial prediction problem.
The class imbalance creates a significant precision-recall trade-off.
The feature-engineering experiments did not produce a meaningful improvement.
The model's recall improvement came with a substantial increase in false positives.
Threshold selection should ideally be treated as a business decision based on the actual costs of false positives and false negatives.
The final test set should be treated as a true holdout and not repeatedly used for model selection.
14. Future Work

The next modeling approach is:

Random Forest

Random Forest will allow us to investigate whether combining many Decision Trees can reduce the weaknesses observed in the single-tree model.

The Random Forest phase will use the Decision Tree as a benchmark and compare:

Accuracy
Precision
Recall
F1
False Positives
False Negatives
Feature Importance
Overfitting
Threshold behavior

Final takeaway

This project was not about finding the highest possible accuracy.

The more important progression was:

"I trained a Decision Tree."
             ↓
"I discovered it was overfitting."
             ↓
"I learned how tree complexity affects generalization."
             ↓
"I learned why test-set tuning is dangerous."
             ↓
"I introduced cross-validation."
             ↓
"I realized accuracy wasn't aligned with the business problem."
             ↓
"I optimized recall."
             ↓
"I investigated class weighting."
             ↓
"I investigated probability thresholds."
             ↓
"I engineered behavioral features."
             ↓
"I tested whether those features actually helped."
             ↓
"I identified the practical limitations of a single Decision Tree."
             ↓
"Now I have a benchmark for Random Forest."
