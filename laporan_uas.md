# Prediksi Kelulusan Mahasiswa Menggunakan Machine Learning Menggunakan Algoritma Decision Tree dan Random Forest

**Nama Kelompok:** Rovi Aulia (2406093) & Rani (NIM)

**Domain Proyek:** Educational Data Mining (Prediksi Kelulusan Mahasiswa)

---

# 1. Judul Proyek

Prediksi Kelulusan Mahasiswa Menggunakan Machine Learning Menggunakan Algoritma Decision Tree dan Random Forest.

Proyek ini bertujuan untuk membangun model klasifikasi yang mampu memprediksi apakah seorang mahasiswa akan berhasil menyelesaikan studinya (**Graduate**) atau **Not Graduate** berdasarkan data akademik, sosial, dan ekonomi yang tersedia pada dataset.

Sesuai ketentuan UAS Kecerdasan Buatan, proyek ini menggunakan **dua algoritma Machine Learning** untuk dibandingkan performanya, yaitu:

- **Model Utama** — Random Forest
- **Model Pembanding** — Decision Tree

Kedua model dibangun menggunakan dataset **Predict Students Dropout and Academic Success** yang diperoleh dari Kaggle. Sebelum proses pelatihan model dilakukan, target pada dataset diubah menjadi dua kelas, yaitu **Graduate** dan **Not Graduate**, sedangkan atribut yang berkaitan dengan nilai Semester 1 dan Semester 2 tidak digunakan agar model lebih berfokus pada karakteristik mahasiswa sebelum memperoleh hasil akademik pada semester tersebut.

Selanjutnya, kedua model dievaluasi menggunakan **Confusion Matrix**, **Accuracy**, **Precision**, **Recall**, dan **F1-Score**. Hasil evaluasi digunakan untuk membandingkan performa kedua algoritma sehingga dapat ditentukan model yang paling baik dalam melakukan prediksi kelulusan mahasiswa.
