import pandas as pd
import numpy as np
#import sklearn
import flwr as fl
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
from tensorflow import keras
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import Parameters, GetParametersRes
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    GetParametersRes,
    Status,
    Code,
)
from typing import Dict, Tuple

# Load dataset
df = pd.read_csv("FEDERATED.csv")

# Encode categorical columns
encoder = LabelEncoder()
df['transaction_type'] = encoder.fit_transform(df['transaction_type'])
df['source_account'] = encoder.fit_transform(df['source_account'])
df['destination_account'] = encoder.fit_transform(df['destination_account'])
df['transaction_mode'] = encoder.fit_transform(df['transaction_mode'])
df['currency'] = encoder.fit_transform(df['currency'])
df['device_type'] = encoder.fit_transform(df['device_type'])
df['region'] = encoder.fit_transform(df['region'])

# Convert timestamp to numeric values (UNIX timestamp)
# Convert 'timestamp' column to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Convert to UNIX timestamp (seconds since 1970)
df["timestamp"] = df["timestamp"].astype("int64") // 10**9

#Normalize Numerical Features
scaler = StandardScaler()
df[['timestamp', 'transaction_amount', 'transaction_frequency', 'balance_before', 'balance_after', 'average_transaction_value']] = scaler.fit_transform(df[['timestamp', 'transaction_amount', 'transaction_frequency', 'balance_before', 'balance_after', 'average_transaction_value']])


# Split into features and target
X = df.drop(columns=["fraud_flag"])
y = df["fraud_flag"]

# Apply SMOTE for balancing
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Save preprocessing objects
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# Split dataset for local training
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define models
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
nb_model = GaussianNB()
knn_model = KNeighborsClassifier(n_neighbors=5)

# CNN Model
features = 13

def create_cnn_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(features,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

cnn_model = create_cnn_model()

class FraudClient(fl.client.NumPyClient):
    def __init__(self, model_rf, model_nb, model_knn, model_cnn, X_train, y_train, X_test, y_test):
        self.model_rf = model_rf
        self.model_nb = model_nb
        self.model_knn = model_knn
        self.model_cnn = model_cnn
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    def get_parameters(self, config):
        return self.model_cnn.get_weights()

    def set_parameters(self, parameters):
        self.model_cnn.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model_cnn.fit(self.X_train, self.y_train, epochs=5, batch_size=32, verbose=0)
        self.model_rf.fit(self.X_train, self.y_train)
        self.model_nb.fit(self.X_train, self.y_train)
        self.model_knn.fit(self.X_train, self.y_train)
        return self.model_cnn.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, accuracy = self.model_cnn.evaluate(self.X_test, self.y_test, verbose=0)
        y_pred_cnn = (self.model_cnn.predict(self.X_test) > 0.5).astype("int32")

        y_pred_rf = self.model_rf.predict(self.X_test)
        y_pred_nb = self.model_nb.predict(self.X_test)
        y_pred_knn = self.model_knn.predict(self.X_test)

        y_pred_hybrid = (y_pred_cnn.flatten() + y_pred_rf + y_pred_nb + y_pred_knn) / 4
        y_pred_final = (y_pred_hybrid > 0.5).astype("int32")

        precision = precision_score(self.y_test, y_pred_final)
        f1 = f1_score(self.y_test, y_pred_final)
        auc = roc_auc_score(self.y_test, y_pred_hybrid)
        logloss = log_loss(self.y_test, y_pred_hybrid)

        print(f"Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Loss: {logloss:.4f}")

        return loss, len(self.X_test), {
            "accuracy": accuracy,
            "precision": precision,
            "f1_score": f1,
            "auc": auc,
            "loss": logloss
        }


def start_client():
    client = FraudClient(rf_model, nb_model, knn_model, cnn_model, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="127.0.0.1:8082", client=client)

start_client()