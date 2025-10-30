import pathlib
import yaml
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def load_data(data_path):
    """Load dataset from CSV"""
    df = pd.read_csv(data_path)
    return df


def split_data(df, test_split, seed):
    """Split dataset into train/test sets"""
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test


def preprocess_data(df):
    """Handle missing values, outliers, encoding, and feature engineering"""
    df = df.copy()

    # Separate target variable if present
    target_col = 'Item_Outlet_Sales'
    if target_col in df.columns:
        Y = df[target_col]
    else:
        Y = None

    # Drop unnecessary columns
    df.drop(columns=['Item_Identifier', 'Item_Outlet_Sales'], errors='ignore', inplace=True)

    # --- Handle Outliers (numeric columns) ---
    for column in df.select_dtypes(include='number').columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower_bound, upper_bound)

    # --- Handle Missing Values ---
    if 'Item_Weight' in df.columns and 'Item_Type' in df.columns:
        df['Item_Weight'] = df.groupby('Item_Type')['Item_Weight'].transform(
            lambda x: x.fillna(x.median())
        )

    if 'Outlet_Size' in df.columns and 'Outlet_Type' in df.columns:
        df['Outlet_Size'] = df.groupby('Outlet_Type')['Outlet_Size'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Medium')
        )

    # --- Standardize Categorical Values ---
    if 'Item_Fat_Content' in df.columns:
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(
            {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}
        )

    # --- Feature Engineering ---
    if 'Outlet_Establishment_Year' in df.columns:
        df['Years_Since_Establishment'] = 2024 - df['Outlet_Establishment_Year']

    # Create visibility bins
    if 'Item_Visibility' in df.columns:
        max_visibility = df['Item_Visibility'].max()
        bins = [-0.001, 0.05, 0.15, max_visibility + 0.001]
        labels = ['Low', 'Medium', 'High']
        df['Item_Visibility_Bins'] = pd.cut(
            df['Item_Visibility'], bins=bins, labels=labels, include_lowest=True
        )

    # Drop Outlet_Identifier if exists
    df.drop('Outlet_Identifier', axis=1, errors='ignore', inplace=True)

    # --- Encoding ---
    nominal_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Type']
    ordinal_columns = ['Item_Visibility_Bins', 'Outlet_Size', 'Outlet_Location_Type']

    # One-Hot Encoding
    nominal_cols_present = [col for col in nominal_columns if col in df.columns]
    if nominal_cols_present:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        encoded_nominal = ohe.fit_transform(df[nominal_cols_present])
        encoded_nominal_df = pd.DataFrame(
            encoded_nominal, columns=ohe.get_feature_names_out(nominal_cols_present)
        )
        df.reset_index(drop=True, inplace=True)
        encoded_nominal_df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, encoded_nominal_df], axis=1)
        df.drop(nominal_cols_present, axis=1, inplace=True)

    # Ordinal Encoding
    ordinal_cols_present = [col for col in ordinal_columns if col in df.columns]
    if ordinal_cols_present:
        ordinal_categories = [
            ['Low', 'Medium', 'High'],     # Item_Visibility_Bins
            ['Small', 'Medium', 'High'],   # Outlet_Size
            ['Tier 1', 'Tier 2', 'Tier 3'] # Outlet_Location_Type
        ]
        ordinal_encoder = OrdinalEncoder(
            categories=ordinal_categories[:len(ordinal_cols_present)],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        df[ordinal_cols_present] = ordinal_encoder.fit_transform(df[ordinal_cols_present])

    # Log transform Item_Visibility if present
    if 'Item_Visibility' in df.columns:
        df['Item_Visibility_Log'] = np.log1p(df['Item_Visibility'])
        df.drop('Item_Visibility', axis=1, inplace=True)

    # Drop original establishment year
    df.drop('Outlet_Establishment_Year', axis=1, errors='ignore', inplace=True)

    # Reattach target if it existed
    if Y is not None:
        df[target_col] = Y.values

    return df


def save_data(train, test, output_path):
    """Save train and test sets"""
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(f"{output_path}/train.csv", index=False)
    test.to_csv(f"{output_path}/test.csv", index=False)


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent  # already fixed earlier
    params_file = home_dir / "params.yaml"
    params = yaml.safe_load(open(params_file))["dataset"]

    input_file = sys.argv[1]
    data_path = (home_dir / input_file).as_posix()  # ✅ FIXED PATH
    output_path = home_dir / "data" / "interim"

    data = load_data(data_path)
    train_data, test_data = split_data(data, params['test_split'], params['seed'])

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    save_data(train_data, test_data, output_path.as_posix())



if __name__ == "__main__":
    main()
