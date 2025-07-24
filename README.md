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

[![Demo Aplikasi](https://i.ytimg.com/vi_webp/tCkaBsp8IBk/0.webp)](https://www.youtube.com/watch?v=tCkaBsp8IBk)  
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
