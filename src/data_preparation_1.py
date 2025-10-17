# Standard library imports
import logging
import re
from typing import Any, Dict

# Related third-party imports
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# In data_preparation.py
class DataPreparation:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = self.create_preprocessor()

    def create_preprocessor(self):
        # Define transformers
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        ordinal_transformer = Pipeline(steps=[("ord", OrdinalEncoder())])
        nominal_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting data cleaning.")
        
        # Drop rows without final_test
        df = df[df['final_test'].notna()]

        # Calculate Q1, Q3, and IQR for 'final_test'
        Q1 = df['final_test'].quantile(0.25)
        Q3 = df['final_test'].quantile(0.75)
        IQR = Q3 - Q1

        # Define upper and lower bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out the outliers
        df = df[(df['final_test'] >= lower_bound) & (df['final_test'] <= upper_bound)]

        # Fill NaN in 'attendance_rate' column with the median '95'.
        df['attendance_rate'] = df['attendance_rate'].fillna(95)

        # For 'CCA', fill the missing rows with 'None'.
        df['CCA'] = df['CCA'].fillna('None')

        # Replace 'SPORTS' with 'Sports', 'ARTS' with 'Arts', 'NONE' with 'None', and 'CLUBS' with 'Clubs' in the 'CCA' column.
        df['CCA'] = df['CCA'].replace('SPORTS', 'Sports')
        df['CCA'] = df['CCA'].replace('ARTS', 'Arts')
        df['CCA'] = df['CCA'].replace('CLUBS', 'Clubs')
        df['CCA'] = df['CCA'].replace('NONE', 'None')

        # Replace 'Y' with 'Yes' and 'N' with 'No' in the 'tuition' column.
        df['tuition'] = df['tuition'].replace('Y', 'Yes')
        df['tuition'] = df['tuition'].replace('N', 'No')

        # Separate the data into features and target
        X = df.drop(columns=['final_test'])
        y = df['final_test']
        
        logging.info("Data cleaning completed.")
        return df

    def create_preprocessor(self) -> ColumnTransformer:
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        nominal_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
        ordinal_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder())])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, ['number_of_siblings', 'hours_per_week', 'attendance_rate', 'n_male', 'n_female', 'age']),
                ("nom", nominal_transformer, ['CCA', 'learning_style', 'gender', 'mode_of_transport', 'bag_color']),
                ("ord", ordinal_transformer, ['direct_admission', 'tuition', 'sleep_time'])
            ]
        )
        return preprocessor
