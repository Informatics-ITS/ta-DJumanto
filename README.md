# 🏁 Tugas Akhir (TA) - Final Project

**Nama Mahasiswa**: Alfa Fakhrur Rizal Zaini  
**NRP**: 5025211214  
**Judul TA**: Deteksi Serangan dan Pencarian Pola pada HTTP  
Log Webserver Menggunakan Pendekatan  
_Bidirectional Long Short-Term Memory_ dengan  
Mekanisme _Attention_   
**Dosen Pembimbing**: Dr. Baskoro Adi P., S.Kom. ,M.Kom   
**Dosen Ko-pembimbing**: Hudan Studiawan, S.Kom., M.Kom. ,Ph.D

---

## 📺 Demo Aplikasi  

[![Demo Aplikasi](https://github-production-user-asset-6210df.s3.amazonaws.com/100863813/470323798-48693a1f-65a7-43c9-934b-22c6346cd40a.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250724%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250724T132230Z&X-Amz-Expires=300&X-Amz-Signature=baa685e5e8dbec9048d5bc1e931da1dbf78e4141bdac036ace1ed8f25899990d&X-Amz-SignedHeaders=host)](https://www.youtube.com/watch?v=tCkaBsp8IBk)  
*Klik gambar di atas untuk menonton demo*

---

## 🛠 Panduan Instalasi & Menjalankan Software  
### Project Structure
```
.
├── LICENSE
├── README.md
├── model
│   ├── ATBiLSTM.py
│   ├── ATBiLSTM_Content.py
│   ├── ATBiLSTM_Structure.py
│   ├── BiLSTM_Content_TFIDF.py
│   ├── BiLSTM_Structure_TFIDF.py
│   ├── BiLSTM_TFIDF.py
│   ├── CNN_Content_TFIDF.py
│   ├── CNN_Structure_TFIDF.py
│   ├── CNN_TFIDF.py
│   ├── LSTM_Content_TFIDF.py
│   ├── LSTM_Structure_TFIDF.py
│   └── LSTM_TFIDF.py
├── pattern_mining
│   ├── all_keyword_vis.py
│   ├── keyword_eps_analysis.py
│   ├── keyword_top_n_analisys.py
│   ├── pattern_min.py
│   ├── pattern_mining.ipynb
│   ├── result_analysis.py
│   ├── sampling.ipynb
│   ├── tfidf
│   │   ├── __pycache__
│   │   │   └── tfidf.cpython-310.pyc
│   │   └── tfidf.py
│   └── visualization_top_n_analysis.py
├── preprocess
│   ├── data_combining.ipynb
│   ├── data_exploration.ipynb
│   ├── labelling.ipynb
│   ├── original datasets.zip
│   └── sampling.ipynb
├── preprocessed datasets
│   ├── csv_dataset_preprocessed_2_balanced.csv
│   └── csv_dataset_preprocessed_2_imbalanced.csv
├── runner
│   ├── ATBILSTM_run.py
│   ├── BILSTM_run.py
│   ├── CNN_run.py
│   ├── LSTM_run.py
│   ├── atbilstm_run.sh
│   ├── bilstm_run.sh
│   ├── cnn_run.sh
│   ├── lstm_run.sh
│   └── outputs
│       ├── attention_outputs
│       │   └── dump.md
│       ├── patterns
│       │   └── dump.md
│       └── predictions
│           └── dump.md
└── tfidf
    └── tfidf.py
```
### Prasyarat  
- Daftar dependensi (contoh):
  - Python 3.9
  - CUDA Driver
  - TFIDF Module
  - Tensorflow
