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

## 2.1 Latar Belakang Permasalahan

Pendidikan tinggi memiliki peran penting dalam menghasilkan sumber daya manusia yang berkualitas dan berdaya saing. Salah satu indikator keberhasilan suatu perguruan tinggi adalah tingkat kelulusan mahasiswa. Mahasiswa yang mampu menyelesaikan studi tepat waktu mencerminkan keberhasilan proses pembelajaran, bimbingan akademik, serta dukungan institusi terhadap kegiatan pendidikan. Sebaliknya, tingginya angka keterlambatan kelulusan maupun mahasiswa yang mengalami *dropout* menjadi tantangan yang perlu mendapatkan perhatian karena dapat memengaruhi kualitas institusi, akreditasi program studi, serta efektivitas penyelenggaraan pendidikan.

Permasalahan kelulusan mahasiswa dipengaruhi oleh berbagai faktor yang saling berkaitan. Faktor akademik seperti prestasi belajar, jumlah mata kuliah yang diambil, serta kemampuan mengikuti proses pembelajaran merupakan faktor yang sering dijadikan indikator keberhasilan studi. Namun demikian, faktor non-akademik juga memiliki pengaruh yang signifikan, seperti kondisi ekonomi keluarga, status beasiswa, usia saat memasuki perguruan tinggi, latar belakang pendidikan sebelumnya, hingga kemampuan mahasiswa dalam memenuhi kewajiban administrasi akademik. Kompleksitas hubungan antar faktor tersebut menyebabkan proses identifikasi mahasiswa yang berpotensi mengalami keterlambatan kelulusan atau *dropout* menjadi tidak mudah apabila hanya dilakukan melalui pengamatan secara manual.

Pada umumnya, pihak perguruan tinggi baru mengetahui adanya permasalahan ketika prestasi akademik mahasiswa mulai menurun atau mahasiswa tidak lagi aktif mengikuti kegiatan perkuliahan. Kondisi tersebut menyebabkan tindakan pendampingan sering kali dilakukan ketika permasalahan sudah berkembang lebih jauh. Padahal, apabila potensi risiko tersebut dapat diketahui lebih awal, institusi dapat melakukan berbagai bentuk intervensi, seperti pemberian bimbingan akademik, konseling, pendampingan belajar, maupun bantuan finansial kepada mahasiswa yang membutuhkan. Oleh karena itu, diperlukan suatu pendekatan yang mampu membantu proses identifikasi mahasiswa yang berpotensi tidak menyelesaikan studi secara lebih cepat, objektif, dan berdasarkan data.

Perkembangan Artificial Intelligence (AI), khususnya pada bidang *Machine Learning*, memberikan peluang untuk menyelesaikan permasalahan tersebut melalui pendekatan berbasis data (*data-driven*). Machine Learning memungkinkan komputer mempelajari pola yang terdapat pada data historis mahasiswa, kemudian menggunakan pola tersebut untuk memprediksi kemungkinan status kelulusan mahasiswa pada data baru. Pendekatan ini tidak hanya membantu meningkatkan efisiensi proses analisis data, tetapi juga memberikan dasar pengambilan keputusan yang lebih objektif dibandingkan dengan penilaian berdasarkan intuisi semata.

Dalam proyek ini digunakan dataset **Predict Students Dropout and Academic Success** yang diperoleh dari Kaggle. Dataset tersebut berisi informasi mengenai karakteristik mahasiswa yang meliputi aspek demografis, sosial, ekonomi, serta akademik. Pada penelitian ini, target klasifikasi disederhanakan menjadi dua kelas, yaitu **Graduate** dan **Not Graduate**. Penyederhanaan kelas dilakukan agar proses klasifikasi lebih sederhana dan lebih mudah diinterpretasikan, sekaligus sesuai dengan tujuan utama penelitian, yaitu membedakan mahasiswa yang berhasil menyelesaikan studi dengan mahasiswa yang belum atau tidak berhasil menyelesaikan studi.

Untuk membangun model prediksi digunakan dua algoritma klasifikasi, yaitu **Decision Tree** dan **Random Forest**. Decision Tree dipilih karena mampu menghasilkan aturan keputusan yang mudah dipahami sehingga proses klasifikasi dapat dijelaskan secara intuitif. Sementara itu, Random Forest dipilih karena merupakan pengembangan dari Decision Tree yang memanfaatkan metode *ensemble learning*, sehingga umumnya mampu menghasilkan performa prediksi yang lebih baik serta lebih tahan terhadap masalah *overfitting*. Kedua algoritma tersebut kemudian dibandingkan menggunakan metrik evaluasi seperti Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix untuk menentukan model yang memberikan performa terbaik dalam memprediksi kelulusan mahasiswa.

## 2.2 Literature Review

Penelitian mengenai prediksi kelulusan maupun *dropout* mahasiswa telah banyak dilakukan dengan memanfaatkan teknik *Machine Learning*. Berbagai algoritma klasifikasi digunakan untuk mengidentifikasi mahasiswa yang berisiko tidak menyelesaikan studi sehingga institusi pendidikan dapat melakukan tindakan pencegahan lebih awal. Berikut merupakan beberapa penelitian yang menjadi referensi dalam proyek ini.

### 1. A Study on Dropout Prediction for University Students Using Machine Learning

Penelitian ini bertujuan untuk membangun model prediksi *dropout* mahasiswa menggunakan beberapa algoritma *Machine Learning*. Dataset yang digunakan berasal dari data akademik mahasiswa yang memuat informasi mengenai karakteristik pribadi, kondisi akademik, dan faktor sosial ekonomi mahasiswa. Penelitian membandingkan beberapa algoritma klasifikasi, seperti Decision Tree, Random Forest, Logistic Regression, Support Vector Machine (SVM), dan XGBoost.

Hasil penelitian menunjukkan bahwa algoritma berbasis *ensemble*, khususnya Random Forest dan XGBoost, memberikan performa yang lebih baik dibandingkan algoritma klasifikasi tunggal. Penelitian ini menunjukkan bahwa pemilihan algoritma yang tepat berpengaruh terhadap tingkat akurasi prediksi mahasiswa yang berpotensi mengalami *dropout*. Oleh karena itu, penelitian tersebut menjadi salah satu dasar pemilihan algoritma Random Forest pada proyek ini.

---

### 2. All-Year Dropout Prediction Modeling and Analysis for University Students

Penelitian ini mengembangkan model prediksi *dropout* mahasiswa yang dapat digunakan pada setiap tahun masa studi mahasiswa. Penelitian tidak hanya berfokus pada mahasiswa tingkat akhir, tetapi juga melakukan analisis terhadap mahasiswa sejak awal masa perkuliahan sehingga proses identifikasi risiko dapat dilakukan lebih dini.

Hasil penelitian menunjukkan bahwa pemanfaatan data historis mahasiswa mampu menghasilkan model prediksi dengan tingkat performa yang baik. Selain itu, penelitian juga menegaskan bahwa informasi akademik dan karakteristik mahasiswa memiliki pengaruh yang signifikan terhadap keberhasilan model klasifikasi. Temuan tersebut mendukung penggunaan dataset yang memiliki atribut akademik dan non-akademik seperti pada proyek ini.

---

### 3. Student Dropout Prediction for University with High Precision and Recall

Penelitian ini berfokus pada peningkatan nilai Precision dan Recall dalam proses prediksi mahasiswa yang berpotensi mengalami *dropout*. Beberapa algoritma *Machine Learning* dibandingkan untuk memperoleh model yang mampu mengurangi kesalahan klasifikasi, terutama pada mahasiswa yang benar-benar berisiko.

Hasil penelitian menunjukkan bahwa evaluasi model tidak cukup hanya menggunakan Accuracy, tetapi juga perlu mempertimbangkan Precision, Recall, dan F1-Score. Oleh karena itu, pada proyek ini evaluasi model dilakukan menggunakan beberapa metrik tersebut agar performa kedua algoritma dapat dibandingkan secara lebih komprehensif.

---

### 4. Predicting Student Dropout Based on Machine Learning and Deep Learning: A Systematic Review

Penelitian ini merupakan *systematic literature review* yang membahas berbagai penelitian mengenai prediksi *dropout* mahasiswa menggunakan metode *Machine Learning* maupun *Deep Learning*. Penelitian mengidentifikasi algoritma yang paling sering digunakan, faktor-faktor yang memengaruhi keberhasilan prediksi, serta tantangan yang masih dihadapi dalam penerapan model prediksi pada dunia pendidikan.

Hasil kajian menunjukkan bahwa Random Forest merupakan salah satu algoritma yang paling sering digunakan karena memiliki performa yang tinggi dan relatif stabil pada berbagai jenis dataset. Selain itu, penelitian juga menyimpulkan bahwa kualitas data dan proses *data preprocessing* memiliki pengaruh besar terhadap performa model yang dihasilkan.

---

### 5. Predictive Analytics Study to Determine Undergraduate Students at Risk of Dropout

Penelitian ini menerapkan teknik *predictive analytics* untuk mengidentifikasi mahasiswa yang memiliki risiko tinggi mengalami *dropout*. Berbagai algoritma klasifikasi dibandingkan berdasarkan performa prediksi menggunakan beberapa metrik evaluasi.

Hasil penelitian menunjukkan bahwa pendekatan berbasis *Machine Learning* mampu membantu institusi pendidikan dalam melakukan identifikasi mahasiswa berisiko secara lebih cepat dibandingkan pendekatan konvensional. Penelitian tersebut juga menunjukkan bahwa model prediksi dapat digunakan sebagai sistem pendukung keputusan (*decision support system*) bagi pihak perguruan tinggi dalam menentukan strategi pendampingan mahasiswa.

---

Berdasarkan kelima penelitian tersebut dapat disimpulkan bahwa penerapan *Machine Learning* dalam bidang pendidikan telah memberikan hasil yang menjanjikan untuk membantu memprediksi keberhasilan studi maupun risiko *dropout* mahasiswa. Meskipun demikian, setiap penelitian menggunakan dataset, atribut, serta kombinasi algoritma yang berbeda-beda sehingga performa model yang dihasilkan juga bervariasi.

Pada proyek ini digunakan dataset **Predict Students Dropout and Academic Success** yang diperoleh dari Kaggle dengan melakukan penyederhanaan target menjadi dua kelas, yaitu **Graduate** dan **Not Graduate**. Selain itu, penelitian ini membandingkan dua algoritma klasifikasi, yaitu **Decision Tree** dan **Random Forest**, untuk mengetahui algoritma yang memberikan performa terbaik dalam memprediksi kelulusan mahasiswa berdasarkan data yang digunakan.

## 2.3 Tujuan Proyek

Berdasarkan permasalahan yang telah diuraikan sebelumnya, proyek ini bertujuan untuk membangun model klasifikasi berbasis *Machine Learning* yang mampu memprediksi status kelulusan mahasiswa menggunakan dataset **Predict Students Dropout and Academic Success**. Model yang dibangun diharapkan dapat mengenali pola dari data historis mahasiswa sehingga mampu mengklasifikasikan mahasiswa ke dalam dua kategori, yaitu **Graduate** dan **Not Graduate**.

Secara khusus, tujuan yang ingin dicapai dalam proyek ini adalah sebagai berikut.

1. Mengolah dataset **Predict Students Dropout and Academic Success** dengan melakukan proses *data preprocessing*, termasuk penyederhanaan target menjadi dua kelas (**Graduate** dan **Not Graduate**) serta menghilangkan atribut yang berkaitan dengan nilai Semester 1 dan Semester 2 agar model tidak bergantung pada hasil akademik pada semester tersebut.

2. Membangun model prediksi menggunakan dua algoritma *Machine Learning*, yaitu **Decision Tree** dan **Random Forest**, untuk melakukan klasifikasi status kelulusan mahasiswa.

3. Membandingkan performa kedua algoritma menggunakan metrik evaluasi **Accuracy**, **Precision**, **Recall**, **F1-Score**, dan **Confusion Matrix** sehingga dapat diketahui algoritma yang memberikan hasil prediksi terbaik.

4. Memberikan gambaran mengenai penerapan *Machine Learning* dalam bidang pendidikan, khususnya sebagai alat bantu untuk mengidentifikasi mahasiswa yang berpotensi mengalami keterlambatan kelulusan atau tidak menyelesaikan studi sehingga institusi pendidikan dapat mengambil langkah pendampingan secara lebih dini.

## 2.4 Pengguna Sistem

Model prediksi yang dibangun pada proyek ini dirancang untuk menghasilkan informasi yang dapat dimanfaatkan oleh berbagai pihak di lingkungan perguruan tinggi. Hasil klasifikasi yang diperoleh dari model *Machine Learning* diharapkan dapat membantu proses pengambilan keputusan yang berkaitan dengan peningkatan tingkat kelulusan mahasiswa. Adapun pengguna yang dapat memanfaatkan hasil prediksi tersebut adalah sebagai berikut.

### 1. Program Studi atau Fakultas

Program studi maupun fakultas dapat memanfaatkan hasil prediksi untuk mengidentifikasi mahasiswa yang memiliki potensi mengalami keterlambatan kelulusan atau tidak menyelesaikan studi. Informasi tersebut dapat digunakan sebagai dasar dalam menyusun program pembinaan akademik, evaluasi kurikulum, maupun penyusunan strategi peningkatan kualitas pendidikan.

### 2. Dosen Wali atau Dosen Pembimbing Akademik

Dosen wali memiliki peran dalam melakukan pendampingan akademik kepada mahasiswa. Dengan adanya hasil prediksi dari model *Machine Learning*, dosen wali dapat lebih mudah menentukan mahasiswa yang memerlukan perhatian khusus sehingga proses pembinaan dapat dilakukan lebih cepat dan lebih tepat sasaran.

### 3. Mahasiswa

Apabila model ini dikembangkan menjadi sebuah sistem informasi atau aplikasi, mahasiswa dapat memanfaatkan hasil prediksi sebagai bahan evaluasi terhadap kondisi akademiknya. Informasi tersebut dapat meningkatkan kesadaran mahasiswa untuk memperbaiki proses belajar, meningkatkan prestasi akademik, maupun memanfaatkan layanan akademik yang disediakan oleh perguruan tinggi.

### 4. Pimpinan Perguruan Tinggi

Pimpinan perguruan tinggi dapat menggunakan hasil prediksi sebagai salah satu pendukung pengambilan keputusan (*decision support*). Informasi mengenai jumlah mahasiswa yang diprediksi berpotensi mengalami keterlambatan kelulusan dapat menjadi dasar dalam menyusun kebijakan akademik, program pembinaan mahasiswa, maupun evaluasi terhadap efektivitas proses pembelajaran yang telah berjalan.

Dengan demikian, model prediksi yang dibangun tidak hanya berfungsi sebagai alat klasifikasi, tetapi juga sebagai pendukung pengambilan keputusan berbasis data (*data-driven decision making*) yang dapat membantu meningkatkan kualitas layanan pendidikan di perguruan tinggi.

## 2.4 Pengguna Sistem

Model prediksi yang dibangun pada proyek ini dirancang untuk menghasilkan informasi yang dapat dimanfaatkan oleh berbagai pihak di lingkungan perguruan tinggi. Hasil klasifikasi yang diperoleh dari model *Machine Learning* diharapkan dapat membantu proses pengambilan keputusan yang berkaitan dengan peningkatan tingkat kelulusan mahasiswa. Adapun pengguna yang dapat memanfaatkan hasil prediksi tersebut adalah sebagai berikut.

### 1. Program Studi atau Fakultas

Program studi maupun fakultas dapat memanfaatkan hasil prediksi untuk mengidentifikasi mahasiswa yang memiliki potensi mengalami keterlambatan kelulusan atau tidak menyelesaikan studi. Informasi tersebut dapat digunakan sebagai dasar dalam menyusun program pembinaan akademik, evaluasi kurikulum, maupun penyusunan strategi peningkatan kualitas pendidikan.

### 2. Dosen Wali atau Dosen Pembimbing Akademik

Dosen wali memiliki peran dalam melakukan pendampingan akademik kepada mahasiswa. Dengan adanya hasil prediksi dari model *Machine Learning*, dosen wali dapat lebih mudah menentukan mahasiswa yang memerlukan perhatian khusus sehingga proses pembinaan dapat dilakukan lebih cepat dan lebih tepat sasaran.

### 3. Mahasiswa

Apabila model ini dikembangkan menjadi sebuah sistem informasi atau aplikasi, mahasiswa dapat memanfaatkan hasil prediksi sebagai bahan evaluasi terhadap kondisi akademiknya. Informasi tersebut dapat meningkatkan kesadaran mahasiswa untuk memperbaiki proses belajar, meningkatkan prestasi akademik, maupun memanfaatkan layanan akademik yang disediakan oleh perguruan tinggi.

### 4. Pimpinan Perguruan Tinggi

Pimpinan perguruan tinggi dapat menggunakan hasil prediksi sebagai salah satu pendukung pengambilan keputusan (*decision support*). Informasi mengenai jumlah mahasiswa yang diprediksi berpotensi mengalami keterlambatan kelulusan dapat menjadi dasar dalam menyusun kebijakan akademik, program pembinaan mahasiswa, maupun evaluasi terhadap efektivitas proses pembelajaran yang telah berjalan.

Dengan demikian, model prediksi yang dibangun tidak hanya berfungsi sebagai alat klasifikasi, tetapi juga sebagai pendukung pengambilan keputusan berbasis data (*data-driven decision making*) yang dapat membantu meningkatkan kualitas layanan pendidikan di perguruan tinggi.

## 2.5 Solusi dan Manfaat Implementasi Artificial Intelligence

Artificial Intelligence (AI), khususnya *Machine Learning*, menawarkan pendekatan yang efektif dalam menyelesaikan permasalahan prediksi kelulusan mahasiswa. Berbeda dengan analisis konvensional yang umumnya dilakukan secara manual, *Machine Learning* mampu mempelajari pola dari data historis mahasiswa sehingga dapat menghasilkan prediksi secara otomatis terhadap data baru.

Pada proyek ini digunakan dua algoritma klasifikasi, yaitu **Decision Tree** dan **Random Forest**. Decision Tree dipilih karena mampu menghasilkan aturan keputusan yang mudah dipahami dan divisualisasikan sehingga proses klasifikasi dapat dijelaskan secara sederhana. Sementara itu, Random Forest digunakan karena merupakan metode *ensemble learning* yang menggabungkan banyak pohon keputusan untuk meningkatkan akurasi prediksi serta mengurangi risiko *overfitting*.

Implementasi kedua algoritma tersebut diharapkan mampu menghasilkan model klasifikasi yang memiliki performa baik dalam membedakan mahasiswa yang diprediksi **Graduate** dan **Not Graduate**. Performa model kemudian dievaluasi menggunakan beberapa metrik, yaitu Accuracy, Precision, Recall, F1-Score, serta Confusion Matrix sehingga dapat diketahui algoritma yang memberikan hasil prediksi terbaik.

Penerapan Artificial Intelligence dalam bidang pendidikan memberikan berbagai manfaat, di antaranya membantu institusi pendidikan dalam mengidentifikasi mahasiswa yang berisiko mengalami keterlambatan kelulusan, meningkatkan efektivitas proses pendampingan akademik, mendukung pengambilan keputusan berbasis data, serta mengoptimalkan pemanfaatan sumber daya yang dimiliki perguruan tinggi. Selain itu, penggunaan model prediksi juga dapat menjadi langkah awal dalam pengembangan sistem *Early Warning System* yang mampu memberikan peringatan dini terhadap mahasiswa yang membutuhkan perhatian khusus selama proses perkuliahan.
