import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """Carga el dataset CSV."""
    df = pd.read_csv(file_path)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def drop_non_predictive_columns(df, columns_to_drop=['Sl_No', 'Customer Key']):
    """Elimina columnas que no aportan información para el clustering."""
    df_clean = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"Columnas eliminadas: {columns_to_drop}")
    return df_clean

def handle_missing_values(df, strategy='median'):
    """Imputa valores nulos con mediana según estrategia."""
    if df.isnull().sum().sum() == 0:
        print("No hay valores nulos.")
        return df
    for col in df.columns:
        if df[col].isnull().any():
            if strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
    print(f"Valores nulos imputados usando {strategy}.")
    return df

def detect_outliers_iqr(df, factor=1.5):
    """Detecta outliers usando IQR y devuelve máscara booleana."""
    outlier_mask = pd.Series(False, index=df.index)
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask |= col_outliers
        print(f"  {col}: {col_outliers.sum()} outliers detectados")
    return outlier_mask

def cap_outliers(df, lower_percentile=1, upper_percentile=99):
    """winsorizar los outliers a percentiles específicos."""
    df_capped = df.copy()
    for col in df_capped.select_dtypes(include=[np.number]).columns:
        lower = np.percentile(df_capped[col], lower_percentile)
        upper = np.percentile(df_capped[col], upper_percentile)
        df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
    print(f"Outliers capped a percentiles {lower_percentile} y {upper_percentile}.")
    return df_capped

def save_cleaned_data(df, output_path='cleaned_data.csv'):
    """Guarda el DataFrame limpio en un archivo CSV."""
    df.to_csv(output_path, index=False)
    print(f"Datos limpios guardados en {output_path}")

def prepare_data(file_path, output_cleaned_path='cleaned_data.csv'):
    """Función principal de toda la preparación."""
    df = load_data(file_path)
    df = drop_non_predictive_columns(df)
    df = handle_missing_values(df)
    # Opcional: ver outliers sin modificar aún
    outlier_mask = detect_outliers_iqr(df)
    print(f"Filas con al menos un outlier: {outlier_mask.sum()}")
    df = cap_outliers(df)
    save_cleaned_data(df, output_cleaned_path)
    return df