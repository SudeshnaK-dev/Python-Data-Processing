import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Load the dataset from a file.
    """
    df = pd.read_csv('creditcard.csv')
    return df

def exploratory_data_analysis(df):
    """
    Perform exploratory data analysis on the dataset.
    This can include visualizations and summary statistics.
    """
    # Example EDA: Display the first few rows of the dataset
    print(df.head())
    # Add more EDA steps as needed

def extract_features_and_target(df):
    """
    Separate the features and the target variable from the dataset.
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):
    """
    Scale the features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_and_predict(X_train_scaled, y_train, X_test_scaled):
    """
    Train a Logistic Regression model and make predictions on the test set.
    """
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return y_pred

def main():
    """
    Main function to load data, perform EDA, split data, scale data, train model, and make predictions.
    """
    print("\t Loading the test data file...")
    df = load_data()

    print("\n\n\t Let's do Exploratory Data Analysis...\n ")
    exploratory_data_analysis(df)

    print("\n\n\t Split the dataset into training and testing sets and train a Logistic Regression Model\n")
    X, y = extract_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    y_pred = train_and_predict(X_train_scaled, y_train, X_test_scaled)

    # Evaluate the model
    print("\n\n\t Model Evaluation\n")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# Example usage
if __name__ == "__main__":
    main()
