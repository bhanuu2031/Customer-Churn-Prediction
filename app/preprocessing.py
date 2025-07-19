import pandas as pd

TRAINING_COLUMNS_PATH = "models/feature_columns.pkl"

def load_and_preprocess(filepath=None, custom_df=None):
    if custom_df is not None:
        df = custom_df.copy()
    else:
        df = pd.read_csv(filepath)

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True)

        df.drop('customerID', axis=1, inplace=True)

        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    if custom_df is None:
        import joblib
        joblib.dump(df.drop("Churn", axis=1).columns.tolist(), TRAINING_COLUMNS_PATH)
        return df

    import joblib
    training_cols = joblib.load(TRAINING_COLUMNS_PATH)
    df = df.reindex(columns=training_cols, fill_value=0)
    return df
