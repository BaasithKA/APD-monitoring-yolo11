# 🛡️ PPE Monitoring System – YOLOv11 Based Computer Vision

Sistem Monitoring Alat Pelindung Diri (APD) otomatis yang dirancang untuk mendeteksi penggunaan perlengkapan keselamatan kerja secara _real-time_. Proyek ini merupakan implementasi dari penelitian berjudul **"Personal Protective Equipment Completeness Monitoring System Using YOLO-Based Computer Vision"**.

## 📝 Deskripsi Proyek

Proyek ini mengembangkan sistem pengawasan otomatis untuk meningkatkan standar Keselamatan dan Kesehatan Kerja (K3) di sektor konstruksi. Dengan memanfaatkan algoritma **YOLOv11s**, sistem ini mampu mengenali tiga jenis APD krusial: helm, masker, dan rompi.

### Kemampuan Utama Sistem:

- **Deteksi Otomatis**: Mengenali pekerja yang menggunakan atau tidak menggunakan APD secara lengkap.
- **Logging & Evidence**: Menyimpan setiap kejadian deteksi ke database SQLite beserta gambar bukti visual sebagai arsip digital.
- **Analytics Dashboard**: Menyajikan data statistik tren kepatuhan pekerja melalui grafik interaktif untuk memudahkan evaluasi manajemen.
- **Ekspor Laporan**: Mendukung pengunduhan riwayat deteksi dalam format CSV untuk kebutuhan audit keselamatan.

## 📊 Manajemen Dataset

Model dilatih menggunakan total **9.202 gambar** yang telah melalui proses augmentasi untuk meningkatkan variasi visual dan mencegah _overfitting_.

### Pembagian Dataset:

- **Train Data (80%)**: 7.362 gambar digunakan untuk pelatihan inti model.
- **Validation Data (15%)**: 1.380 gambar digunakan untuk validasi selama proses _training_.
- **Test Data (5%)**: 460 gambar (_unseen data_) digunakan untuk evaluasi akhir performa sistem.

Dataset tersedia di platform Roboflow:
👉 [**Dataset APD Dieet - Roboflow Universe**](https://universe.roboflow.com/apd-dieet/apd-goksf)

## 🚀 Performa Model (YOLOv11s)

Berdasarkan evaluasi pada _test set_ independen, model **YOLOv11s** menunjukkan performa yang sangat efisien dan akurat:

| Metrik Evaluasi     | Hasil (YOLOv11s) |
| :------------------ | :--------------: |
| **Presisi (P)**     |       0.92       |
| **Recall (R)**      |       0.86       |
| **mAP@0.5**         |      0.906       |
| **mAP@0.5:0.95**    |      0.544       |
| **Waktu Inferensi** |      8.9 ms      |

Sistem ini mampu memproses video dengan kecepatan mencapai **~112 FPS**, menjadikannya sangat andal untuk pemantauan keamanan secara langsung tanpa jeda yang signifikan.

## 🛠️ Tech Stack

- **AI & Backend**: Python, Flask, SQLite3, OpenCV, Ultralytics YOLOv11s.
- **Frontend**: Bootstrap 5, Chart.js, Bootstrap Icons.
- **Tools**: Roboflow (Annotation), Google Colab (Training GPU A100).

## 📁 Struktur Folder Proyek

```text
backend/
|-- static/
|   `-- processed/      # Penyimpanan gambar bukti (.jpg)
|-- templates/          # File frontend (index, history, dashboard)
|-- app.py              # Logika utama aplikasi & Inference
|-- best.pt             # Bobot model YOLOv11s terbaik
|-- deteksi.db          # Database SQLite
|-- init_db.py          # Skrip inisialisasi tabel database
`-- requirements.txt    # Daftar dependensi library
```
⚙️ Cara Menjalankan Sistem
1. Instalasi
Pastikan Anda telah menginstal Python versi 3.9 atau lebih baru.

Bash
# Install library yang diperlukan
pip install -r requirements.txt

# Inisialisasi database awal
python init_db.py
2. Menjalankan Aplikasi
Bash
# Jalankan server Flask
python app.py
Akses aplikasi melalui browser di: http://127.0.0.1:5000
```
