import tensorflow as tf
import numpy as np

# 1. Load otak cerdas yang sudah kita simpan tadi
model = tf.keras.models.load_model('/workspace/model_node1_final.h5')

def deteksi_paket(fitur_jaringan):
    # fitur_jaringan adalah data paket yang tertangkap oleh GNS3
    # Ubah menjadi format yang sama dengan yang dipakai saat training
    input_data = np.array(fitur_jaringan).reshape(1, -1)
    
    # AI menebak: Normal, DDoS, atau PortScan?
    prediksi = model.predict(input_data)
    hasil = np.argmax(prediksi)
    
    label_map = {0: 'BENIGN', 1: 'DDOS', 2: 'PORTSCAN'}
    return label_map[hasil]

# Contoh penggunaan:
# print(deteksi_paket([0.1, 0.5, ...]))