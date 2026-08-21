import flwr as fl
import tensorflow as tf
import pandas as pd
import sys
import numpy as np
import json
import time

node_id = sys.argv[1]

dataset_map = {
    "1": "dataset_node1_ddos.csv",
    "2": "dataset_node2_portscan.csv",
    "3": "dataset_node3_benign.csv"
}

file_path = f"/workspace/{dataset_map[node_id]}"
df = pd.read_csv(file_path)

# Rapikan nama kolom
df.columns = df.columns.str.strip()

# Pastikan ada kolom Label untuk training
if "Label" not in df.columns:
    raise ValueError("Dataset training wajib punya kolom Label.")

# Pisahkan fitur dan label
feature_columns = [col for col in df.columns if col != "Label"]

X_df = df[feature_columns].copy()
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df = X_df.replace([np.inf, -np.inf], np.nan)
X_df = X_df.fillna(0)

X = X_df.values.astype("float32")

# Normalisasi
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / (X_std + 1e-7)

# Simpan scaler untuk deteksi CSV baru
scaler_data = {
    "mean": X_mean.tolist(),
    "std": X_std.tolist(),
    "labels": ["BENIGN", "DDOS", "PORTSCAN"]
}

with open(f"/workspace/scaler_node{node_id}.json", "w") as f:
    json.dump(scaler_data, f)

with open(f"/workspace/feature_columns_node{node_id}.json", "w") as f:
    json.dump(feature_columns, f)

# Rapikan label
df["Label"] = df["Label"].astype(str).str.strip().str.upper()

kategori_global = pd.CategoricalDtype(
    categories=["BENIGN", "DDOS", "PORTSCAN"]
)

df["Label"] = df["Label"].astype(kategori_global)

y = pd.get_dummies(df["Label"])
y = y.reindex(columns=["BENIGN", "DDOS", "PORTSCAN"], fill_value=0)
y = y.values.astype("float32")

# Model AI
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


class CIDSClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)

        print("=" * 60)
        print(f"[NODE {node_id}] Mulai training lokal")
        print(f"[NODE {node_id}] Jumlah data: {len(X)}")
        print("=" * 60)

        model.fit(X, y, epochs=5, batch_size=32, verbose=1)

        print("=" * 60)
        print(f"[NODE {node_id}] Training lokal selesai")
        print("=" * 60)

        return model.get_weights(), len(X), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)

        loss, acc = model.evaluate(X, y, verbose=0)

        nama_file = f"/workspace/model_node{node_id}_final.h5"
        model.save(nama_file)

        print("=" * 60)
        print(f"[NODE {node_id}] Evaluasi selesai")
        print(f"[NODE {node_id}] Loss     : {loss:.4f}")
        print(f"[NODE {node_id}] Accuracy : {acc:.4f}")
        print(f"[NODE {node_id}] Model disimpan ke: {nama_file}")
        print("=" * 60)

        return loss, len(X), {"accuracy": float(acc)}


# Tunggu server siap
time.sleep(5)

fl.client.start_numpy_client(
    server_address="server:8081",
    client=CIDSClient()
)