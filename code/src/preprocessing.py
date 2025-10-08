import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_data(path):
    df = pd.read_csv(path)
    return df

def build_preprocessor():
    # Ordinal mapping for education
    education_order = ['unknown', 'primary', 'secondary', 'tertiary']
    ordinal_cols = ['education']
    nominal_cols = ['job', 'marital', 'contact', 'month', 'poutcome']
    binary_cols = ['default', 'housing', 'loan']
    numeric_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'day']

    ordinal_pipe = Pipeline([
        ('ord', OrdinalEncoder(categories=[education_order], handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    nominal_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    binary_pipe = Pipeline([
        ('onehot', OneHotEncoder(drop='if_binary'))
    ])
    numeric_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('ord', ordinal_pipe, ordinal_cols),
        ('nom', nominal_pipe, nominal_cols),
        ('bin', binary_pipe, binary_cols),
        ('num', numeric_pipe, numeric_cols)
    ])
    return preprocessor

def preprocess(df, preprocessor=None, fit=True):
    X = df.drop(['y', 'id'], axis=1)
    y = df['y']
    # Optionally create interaction features here (example)
    X['balance_x_duration'] = X['balance'] * X['duration']
    X['age_x_education'] = X['age'] * X['education'].map({'unknown':0, 'primary':1, 'secondary':2, 'tertiary':3})
    if fit:
        X_processed = preprocessor.fit_transform(X)
    else:
        X_processed = preprocessor.transform(X)
    return X_processed, y

if __name__ == "__main__":
    #df = load_data("../data/train.csv")
    df = load_data(r"D:\GUVI-Projects\Capstone\Project1_Bank\tmp\proj\files (1)\data\train.csv")
    preprocessor = build_preprocessor()
    X, y = preprocess(df, preprocessor)