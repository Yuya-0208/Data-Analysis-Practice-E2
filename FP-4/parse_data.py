import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    
    df0 = pd.read_csv(file_path)
    
    df = df0[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Delicassen']].copy()

    df = df.rename(
    columns={
            'Fresh': 'Fre',
            'Grocery': 'Gro',
            'Frozen': 'Fro',
            'Delicassen': 'Deli'
        }
    )

    return df

def extract_variables(df):
    
    Fre = df['Fre']
    Milk = df['Milk']
    Gro = df['Gro']
    Fro = df['Fro']
    Deli = df['Deli']

    return Fre, Milk, Gro, Fro, Deli