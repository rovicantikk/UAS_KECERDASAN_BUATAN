# Prediksi Kelulusan Mahasiswa Menggunakan Machine Learning dengan Algoritma Decision Tree dan Random Forest

## Deskripsi Proyek

Proyek ini merupakan tugas UAS mata kuliah **Kecerdasan Buatan** yang bertujuan membangun model *Machine Learning* untuk memprediksi status kelulusan mahasiswa menggunakan dataset **Predict Students Dropout and Academic Success** dari Kaggle.

Penelitian menggunakan dua algoritma klasifikasi, yaitu **Decision Tree** dan **Random Forest**, kemudian membandingkan performanya berdasarkan metrik evaluasi seperti Accuracy, Precision, Recall, dan F1-Score.

---

## Tujuan Proyek

- Memprediksi status kelulusan mahasiswa.
- Membandingkan performa algoritma Decision Tree dan Random Forest.
- Menentukan algoritma dengan performa terbaik berdasarkan hasil evaluasi.

---

## Dataset

- **Nama Dataset** : Predict Students Dropout and Academic Success
- **Sumber** : Kaggle
- **Jumlah Data** : 4.424
- **Jumlah Atribut Awal** : 37
- **Target Awal** :
  - Graduate
  - Dropout
  - Enrolled
- **Target Akhir** :
  - Graduate
  - Not Graduate

---

## Algoritma yang Digunakan

- Decision Tree
- Random Forest

---

## Tahapan Pengerjaan

### 1. Business Understanding

Menentukan permasalahan yang akan diselesaikan, tujuan penelitian, manfaat implementasi Artificial Intelligence, serta melakukan kajian pustaka berdasarkan jurnal ilmiah yang relevan.

### 2. Data Understanding

Mengumpulkan dataset dari Kaggle, mempelajari karakteristik data, serta mengidentifikasi atribut yang akan digunakan dalam proses pemodelan.

### 3. Exploratory Data Analysis (EDA)

Melakukan eksplorasi data menggunakan visualisasi untuk mengetahui distribusi data, hubungan antar variabel, serta memperoleh gambaran awal mengenai karakteristik dataset.

### 4. Data Preparation

Melakukan preprocessing data dengan langkah-langkah berikut:

- Mengubah target menjadi dua kelas (Graduate dan Not Graduate).
- Menghapus atribut Semester 1 dan Semester 2.
- Melakukan Label Encoding.
- Membagi data menjadi data latih dan data uji.

### 5. Modeling

Membangun model menggunakan algoritma Decision Tree dan Random Forest.

### 6. Evaluation

Melakukan evaluasi menggunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Kemudian membandingkan performa kedua model.

---

## Hasil

Hasil penelitian menunjukkan bahwa algoritma **Random Forest** memberikan performa yang lebih baik dibandingkan Decision Tree.

| Algoritma | Accuracy |
|-----------|----------|
| Decision Tree | 75,48% |
| Random Forest | **83,84%** |

---

## Struktur Repository

```text
UAS-KecerdasanBuatan/
├── README.md
├── Laporan_uas.md
├── uas_model.ipynb
├── image/
└── data/
    ├── dataset/
    └── jurnal/
```

---

## Penulis

- **Rovi Aulia** (2406093)
- **Rani Nurcahyani** (2406103)
