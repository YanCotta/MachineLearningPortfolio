# Process Documentation

## Dataset
- Filename: Social_Network_Ads.csv
- Columns: Age, EstimatedSalary, Purchased (0 or 1)

## Steps
1. Load the CSV and split into features (X) and target (y).
2. Perform train_test_split to create training and test sets.
3. For most scripts, apply StandardScaler to both sets.
4. Train the classifier, predict on test set.
5. Evaluate using confusion matrix and accuracy score.

## Scripts
- logistic_regression.py  
- kernel_svm.py  
- naive_bayes.py  
- random_forest_classification.py  
- k_nearest_neighbors.py  
- decision_tree_classification.py  
- support_vector_machine.py

Use any of these scripts to train a classification model on Social_Network_Ads.csv.
