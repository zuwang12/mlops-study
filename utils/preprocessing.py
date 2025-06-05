import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(train_df, test_df, target_col='채무 불이행 여부'):
    y = train_df[target_col]
    X = train_df.drop(columns=['UID', target_col])
    X_test = test_df.drop(columns=['UID'])

    cat_cols = X.select_dtypes(include='object').columns
    encoders = {col: LabelEncoder().fit(pd.concat([X[col], X_test[col]])) for col in cat_cols}
    for col in cat_cols:
        X[col] = encoders[col].transform(X[col])
        X_test[col] = encoders[col].transform(X_test[col])

    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler().fit(X[num_cols])
    X[num_cols] = scaler.transform(X[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X, y, X_test
