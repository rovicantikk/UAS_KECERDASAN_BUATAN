# Prediksi Kelulusan Mahasiswa Menggunakan Algoritma Decision Tree dan Random Forest Berbasis Machine Learning

**Nama Kelompok:** Rovi Aulia (2406093) & Rani Nurcahyani (2406103)

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

# 2. Business Understanding

## 2.1 Permasalahan Dunia Nyata
Kelulusan mahasiswa merupakan salah satu indikator penting dalam menilai kualitas penyelenggaraan pendidikan di perguruan tinggi. Tingkat kelulusan yang tinggi menunjukkan bahwa proses pembelajaran, bimbingan akademik, serta layanan pendidikan berjalan dengan baik. Sebaliknya, tingginya angka mahasiswa yang mengalami keterlambatan kelulusan atau bahkan *dropout* menjadi tantangan yang harus diperhatikan oleh institusi pendidikan.

Berbagai faktor dapat memengaruhi keberhasilan mahasiswa dalam menyelesaikan studinya. Faktor-faktor tersebut tidak hanya berasal dari aspek akademik, seperti prestasi belajar dan jumlah mata kuliah yang ditempuh, tetapi juga dipengaruhi oleh kondisi sosial, ekonomi, usia saat masuk perguruan tinggi, status beasiswa, hingga kemampuan mahasiswa dalam memenuhi kewajiban administrasi akademik. Kompleksitas faktor tersebut menyebabkan pihak perguruan tinggi sering mengalami kesulitan dalam mengidentifikasi mahasiswa yang berisiko tidak menyelesaikan studinya secara tepat waktu.

Apabila potensi kegagalan studi dapat diprediksi sejak dini, perguruan tinggi dapat memberikan berbagai bentuk intervensi, seperti bimbingan akademik, konseling, bantuan finansial, maupun pendampingan belajar kepada mahasiswa yang membutuhkan. Dengan demikian, tingkat keberhasilan studi mahasiswa diharapkan dapat meningkat sekaligus menurunkan angka *dropout*.

Perkembangan teknologi Artificial Intelligence (AI), khususnya Machine Learning, memberikan peluang untuk membantu menyelesaikan permasalahan tersebut. Melalui teknik klasifikasi, model Machine Learning dapat mempelajari pola dari data historis mahasiswa dan menghasilkan prediksi terhadap status kelulusan mahasiswa berdasarkan karakteristik yang dimiliki. Informasi hasil prediksi tersebut dapat menjadi dasar bagi pengambilan keputusan yang lebih cepat dan lebih objektif oleh pihak perguruan tinggi.
