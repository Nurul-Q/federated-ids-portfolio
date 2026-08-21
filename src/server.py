import flwr as fl
from flwr.server.strategy import FedAvg

# Fungsi agar Mandor (Server) mau menghitung rata-rata akurasi dari semua Node
def hitung_akurasi_global(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Masukkan fungsi hitung akurasi ke dalam Strategi Server
strategy = FedAvg(
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=hitung_akurasi_global, # <-- Ini mantra ajaibnya!
)

# Memulai Server
fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
