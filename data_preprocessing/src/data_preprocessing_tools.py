"""
Tools for loading data, handling missing values, encoding categorical features,
splitting into training/testing sets, and scaling numeric features.
"""
import numpy as np
import pandas as pd

def main():
    """
    Main function demonstrating usage of data preprocessing steps.
    """
    # 1. Load the dataset (fill in your file path)
    data_set = pd.read_csv('your_dataset.csv')

    # 2. Separate your features (X) and target (y)
    # Adjust the indexing as needed
    X = data_set.iloc[:, :-1].values
    y = data_set.iloc[:, -1].values

    # 3. Handle missing values (use appropriate columns and strategy)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])

    # 4. Encode categorical features (update columns to encode)
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    # 5. Encode target if it's categorical
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 6. Split into training and testing data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # 7. Feature scaling (adjust column range as needed)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
    X_test[:, 3:] = sc.transform(X_test[:, 3:])

if __name__ == "__main__":
    main()