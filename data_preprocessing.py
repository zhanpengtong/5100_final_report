import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Convert 'Blood Pressure' from '126/83' to two separate columns 'Systolic' and 'Diastolic'
    blood_pressure = data['Blood Pressure'].str.split('/', expand=True)
    data['Systolic'] = pd.to_numeric(blood_pressure[0], errors='coerce')
    data['Diastolic'] = pd.to_numeric(blood_pressure[1], errors='coerce')
    data.drop(columns=['Blood Pressure'], inplace=True)  # Remove the original non-numeric column

    # Select features and target
    X = data.drop('Sleep Disorder', axis=1)
    y = data['Sleep Disorder']

    # Define numeric and categorical features
    numeric_features = X.select_dtypes(include=['int', 'float']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Numeric transformation pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # Categorical transformation pipeline (adjust as necessary for your data)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Preprocessor to handle all columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough')

    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example file path
    X_train, X_test, y_train, y_test = preprocess_data('path_to_your_data.csv')
    print("Preprocessing completed. Data shapes:", X_train.shape, X_test.shape)
