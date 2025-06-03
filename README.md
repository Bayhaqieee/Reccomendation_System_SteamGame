# Laporan Proyek Machine Learning - Muhammad Aditya Bayhaqie

## Project Overview

### Domain Proyek
Proyek ini berada dalam domain Sistem Rekomendasi (Recommender Systems), khususnya fokus pada pembuatan sistem rekomendasi konten untuk game di platform Steam.

### Masalah yang Dihadapi
Pengguna di platform game seperti Steam seringkali kesulitan menemukan game baru yang sesuai dengan preferensi mereka di tengah banyaknya pilihan game yang tersedia. Ini menyebabkan potensi hilangnya kesempatan bagi pengguna untuk menemukan game yang menarik dan juga bagi pengembang game untuk menjangkau audiens yang relevan.

### Latar Belakang Pemilihan Masalah
Pemilihan masalah ini didasarkan pada pengamatan bahwa rekomendasi game yang efektif sangat penting dalam meningkatkan pengalaman pengguna dan keberhasilan distribusi game. Sistem rekomendasi konten sangat relevan ketika data interaksi pengguna (seperti rating spesifik per game per user) terbatas, namun deskripsi dan fitur game (konten) tersedia secara melimpah. Dataset Steam Games di Kaggle menyediakan data konten yang kaya, menjadikannya cocok untuk pendekatan Content-Based Filtering.

### Solusi
Solusi yang diusulkan adalah membangun sistem rekomendasi game berbasis konten menggunakan dua pendekatan:
1.  **Content-Based Filtering Tradisional:** Menggunakan TF-IDF Vectorizer untuk merepresentasikan fitur teks game dan Cosine Similarity untuk mengukur kemiripan antar game.
2.  **Deep Content Filtering:** Menggunakan model neural network (mirip arsitektur RecommenderNet, diadaptasi untuk data tanpa user) untuk mempelajari representasi (embedding) game dari fitur kontennya, kemudian menghitung kemiripan antar embedding.

## Business Understanding

### Problem Statements
*   Bagaimana cara merekomendasikan game kepada pengguna berdasarkan fitur-fitur game itu sendiri (seperti genre, tag, kategori, dll.)?
*   Bagaimana membandingkan kinerja dua pendekatan berbasis konten yang berbeda (tradisional vs. deep learning) untuk rekomendasi game?
*   Bagaimana mengevaluasi kualitas rekomendasi yang diberikan oleh sistem, baik secara kuantitatif maupun kualitatif?

### Goals
*   Membuat sistem rekomendasi game menggunakan Content-Based Filtering tradisional (TF-IDF + Cosine Similarity).
*   Membangun model Deep Learning untuk menghasilkan embedding game dari fitur kontennya dan menggunakannya untuk rekomendasi.
*   Melakukan evaluasi terhadap kedua model rekomendasi.
*   Menyajikan rekomendasi game berdasarkan masukan nama game tertentu.

### Solution Statements
Kami akan mengembangkan dua model sistem rekomendasi berbasis konten. Model pertama akan menggunakan TF-IDF untuk vektorisasi teks dan Cosine Similarity. Model kedua akan memanfaatkan arsitektur deep learning untuk mempelajari representasi game yang lebih kompleks. Kedua model akan dievaluasi berdasarkan kemampuannya memberikan rekomendasi game yang relevan, diukur melalui evaluasi kualitatif (dengan simulasi relevansi menggunakan skor kemiripan dan fitur) dan potensi evaluasi kuantitatif di masa depan jika data interaksi tersedia.

## Data Understanding

### URL/Tautan Sumber Data
Dataset: [Steam Games Dataset – Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data?select=games.csv)

### Jumlah Baris dan Kolom
Dataset awal memiliki 111452 baris dan 39 kolom. Setelah proses persiapan data (penanganan missing values, drop kolom, drop duplikat), DataFrame `df` yang digunakan untuk proses selanjutnya memiliki 110326 baris dan 27 kolom. DataFrame `games_df` yang digunakan untuk pemodelan (setelah seleksi fitur dan pengurangan ukuran data menjadi 20%) memiliki sekitar 22000 baris dan 11 kolom (sebelum penambahan fitur hasil normalisasi Categories dan Tags).

### Kondisi dan Karakteristik Data
*   Dataset berisi informasi detail tentang game-game di Steam.
*   Terdapat banyak kolom dengan nilai kosong (NaN) yang memerlukan penanganan.
*   Beberapa kolom (seperti `Reviews`, `Metacritic url`, `Score rank`) memiliki persentase missing value yang sangat tinggi.
*   Beberapa kolom teks (`About the game`, `Supported languages`, `Full audio languages`, `Categories`, `Tags`) memerlukan pembersihan dan normalisasi.
*   Kolom `Release date` perlu diubah ke tipe data datetime.
*   Terdapat data duplikat berdasarkan nama game.
*   Fitur teks seperti `Categories` dan `Tags` merupakan string terformat (dipisahkan koma) yang perlu diparse.

### Fitur dalam Dataset
Fitur-fitur kunci yang digunakan dalam proyek ini (setelah seleksi dan rekayasa fitur):
*   `Name`: Nama game (Identifikasi unik dan target rekomendasi).
*   `Release date`: Tanggal rilis game.
*   `Required age`: Batas usia yang disarankan.
*   `Supported languages`: Jumlah bahasa yang didukung.
*   `Full audio languages`: Jumlah bahasa audio penuh yang didukung.
*   `Windows`: Dukungan platform Windows (Boolean/Encoded).
*   `Mac`: Dukungan platform Mac (Boolean/Encoded).
*   `Linux`: Dukungan platform Linux (Boolean/Encoded).
*   `Average playtime forever`: Rata-rata total waktu bermain.
*   `Player based`: Orientasi pemain (Single, Multi).
*   `Steam Achievements`: Dukungan achievement Steam (Boolean/Encoded).
*   `Family Sharing`: Dukungan Family Sharing (Boolean/Encoded).
*   `Full controller support`: Dukungan kontroler penuh (Boolean/Encoded).
*   `Tag 1`: Tag utama game (hasil ekstraksi).
*   `Tag 2`: Tag kedua game (hasil ekstraksi).
*   `Tag 3`: Tag ketiga game (hasil ekstraksi).

## Data Preparation
Tahapan persiapan data meliputi:
1.  **Loading Data:** Membaca file `games.csv` ke dalam pandas DataFrame.
2.  **Data Assessment:** Memeriksa informasi dasar dataset (`info()`, `head()`, `isnull().sum()`) untuk memahami struktur, tipe data, dan keberadaan missing values. Mengidentifikasi kolom yang perlu di-drop atau diisi. Melakukan rename kolom awal yang teridentifikasi salah.
3.  **Missing Value Treatment:**
    *   Drop baris dengan `Name` atau `Developers` yang null.
    *   Mengisi `About the game` dengan 'No Description'.
    *   Mengisi `Publishers` yang null dengan nilai dari `Developers`.
    *   Drop baris dengan `Categories` atau `Genres` atau `Release date` yang null.
    *   Mengisi `Tags` yang null dengan nilai dari `Genres`.
    *   Drop kolom yang tidak relevan atau memiliki banyak missing value (`Reviews`, `Website`, `Support url`, `Support email`, `Metacritic url`, `Metacritic score`, `Score rank`, `Notes`, `Average playtime two weeks`, `Median playtime two weeks`, `Screenshots`, `Movies`, `Header image`).
4.  **Data Type Modification:** Mengubah kolom `Release date` ke tipe data datetime.
5.  **Value Modification/Normalization:** Mengganti nilai '[]' pada `Supported languages` dan `Full audio languages` menjadi string 'No Supported languages' atau 'No Full audio languages'.
6.  **Dropping Duplicates:** Menghapus baris duplikat berdasarkan nama game.
7.  **Data Selection and Conversion:** Memilih kolom-kolom yang relevan dan mengubahnya menjadi Python lists.
8.  **Dictionary Making & Reduction:** Membuat DataFrame baru (`games_df`) dari list yang dipilih dan mengurangi ukurannya menjadi 20% dari data asli untuk eksperimen yang lebih cepat.
9.  **Feature Engineering/Normalization:**
    *   Memproses kolom `Categories` untuk mengekstrak `Player based`, `Steam Achievements`, `Family Sharing`, `Full controller support`.
    *   Memproses kolom `Tags` untuk mengekstrak `Tag 1`, `Tag 2`, `Tag 3`.
    *   Menghitung jumlah bahasa pada `Supported languages` dan `Full audio languages`.
    *   Mendrop kolom `Categories` dan `Tags` asli.
10. **Data Randomizing:** Mengacak urutan baris pada `games_df`.

## Modeling

### Model 1: TF-IDF + Cosine Similarity
*   **Teknik:** Content-Based Filtering berbasis representasi teks.
*   **Langkah-langkah:**
    *   Menggabungkan fitur-fitur yang relevan (`Release date`, `Required age`, `Supported languages`, dll., termasuk fitur hasil rekayasa) menjadi satu string teks per game.
    *   Menggunakan `TfidfVectorizer` dari scikit-learn untuk mengubah string teks tersebut menjadi matriks fitur TF-IDF. TF-IDF memberikan bobot pada kata berdasarkan frekuensinya dalam dokumen dan invers frekuensinya di seluruh korpus.
    *   Menghitung matriks kemiripan kosinus (`cosine_similarity`) antara semua pasangan vektor TF-IDF game. Skor kemiripan kosinus bernilai antara 0 dan 1, menunjukkan seberapa mirip konten dua game.
    *   Membuat fungsi `get_recommendations` yang menerima nama game sebagai input, mencari indeks game tersebut, mengambil baris kemiripannya dari matriks kemiripan kosinus, mengurutkan game lain berdasarkan skor kemiripan, dan mengembalikan N game teratas yang paling mirip.

### Model 2: Deep Content Filtering
*   **Teknik:** Content-Based Filtering berbasis embedding yang dipelajari oleh neural network.
*   **Langkah-langkah:**
    *   Mengidentifikasi fitur kategorikal (`Player based`, `Tag 1`, `Tag 2`, `Tag 3`, dll.) dan fitur numerik (`Required age`, `Supported languages`, dll.).
    *   Melakukan Label Encoding pada fitur kategorikal.
    *   Melakukan penskalaan (StandarScaler) pada fitur numerik.
    *   Membangun model Sequential atau Functional API menggunakan Keras (TensorFlow).
    *   Membuat `Input` layer terpisah untuk setiap fitur.
    *   Menggunakan `Embedding` layer untuk fitur kategorikal untuk mempelajari representasi kepadatan (dense representation). Dimensi embedding ditentukan secara heuristik.
    *   Menggunakan `Flatten` layer setelah embedding.
    *   Menggabungkan (concatenate) output dari semua embedding layer dan input numerik.
    *   Menambahkan beberapa `Dense` layer dengan aktivasi ReLU dan `Dropout` untuk mempelajari pola yang kompleks.
    *   Menambahkan `Dense` layer terakhir (output layer) dengan aktivasi linear untuk menghasilkan vektor embedding game (misalnya, 32 dimensi).
    *   Melakukan prediksi pada data game yang telah di-encode untuk mendapatkan matriks embedding game.
    *   Menghitung matriks kemiripan kosinus (`cosine_similarity`) antara semua pasangan vektor embedding game.
    *   Membuat fungsi `get_deep_content_based_recommendations` yang mirip dengan model pertama, tetapi menggunakan matriks kemiripan kosinus dari embedding deep learning.

## Evaluation

### Evaluasi Kuantitatif
Pada proyek ini, evaluasi kuantitatif tradisional seperti Precision, Recall, atau F1-score (yang memerlukan data interaksi pengguna yang relevan) tidak dapat dilakukan secara langsung karena keterbatasan dataset yang hanya berisi data konten. Namun, metrik yang dapat diamati dari proses pemodelan adalah:
*   **Ukuran Matriks TF-IDF dan Cosine Similarity:** Menunjukkan dimensi ruang fitur.
*   **Bentuk Matriks Embedding Game:** Menunjukkan dimensi representasi yang dipelajari oleh model deep learning.
*   **Skor Kemiripan Kosinus:** Nilai numerik ini sendiri bisa dianggap sebagai metrik kuantitatif dari seberapa mirip dua item berdasarkan model.

### Evaluasi Kualitatif
Evaluasi kualitatif dilakukan dengan cara:
1.  **Memeriksa Rekomendasi:** Memilih game tertentu dan melihat daftar game yang direkomendasikan oleh kedua model. Secara manual menilai apakah game yang direkomendasikan tampak relevan berdasarkan nama, genre, tag, dan fitur-fitur lain dari game masukan.
2.  **Menganalisis Fitur Game Masukan dan Rekomendasi:** Membandingkan fitur-fitur game masukan dengan fitur-fitur game yang direkomendasikan untuk memahami dasar kemiripan yang ditemukan oleh model.
3.  **Simulasi Relevansi:** Menggunakan simulasi sederhana untuk menghitung "Precision@N". Dalam simulasi ini, sebuah rekomendasi dianggap "relevan" jika skor kemiripannya di atas ambang batas tertentu (misalnya, > 0.7 untuk TF-IDF, > 0.6 untuk Deep) DAN (opsional) berbagi setidaknya satu tag utama (`Tag 1`) dengan game masukan. Ini adalah simulasi dan bukan evaluasi yang sebenarnya dengan data relevansi pengguna. Metrik ini memberikan perkiraan kasar tentang seberapa sering model merekomendasikan game yang "mirip" berdasarkan definisi kemiripan simulasi kita.

## Kesimpulan
*   Kedua model, baik TF-IDF + Cosine Similarity maupun Deep Content Filtering, berhasil dibangun dan mampu menghasilkan rekomendasi game berdasarkan konten.
*   Model TF-IDF + Cosine Similarity memberikan rekomendasi berdasarkan kemiripan literal dari fitur-fitur yang digabungkan menjadi teks. Ini cenderung menangkap kemiripan berdasarkan keberadaan kata kunci fitur.
*   Model Deep Content Filtering berusaha mempelajari representasi game dalam ruang embedding. Ini berpotensi menangkap pola kemiripan yang lebih kompleks dan non-linear antar fitur dibandingkan metode tradisional.
*   Evaluasi kualitatif menunjukkan bahwa kedua model menghasilkan game yang *tampak* relevan berdasarkan fitur-fiturnya.
*   Simulasi Precision@N memberikan indikasi kasar tentang seberapa sering rekomendasi dianggap "relevan" berdasarkan ambang batas skor kemiripan. Hasil simulasi ini sangat bergantung pada ambang batas yang dipilih dan definisi relevansi simulasi.

## Rekomendasi
*   **Peningkatan Data:** Jika data interaksi pengguna (playtime per user, rating, review spesifik user) tersedia, gabungkan data konten dengan data interaksi untuk membangun model Hybrid Recommender System (misalnya, Collaborative Filtering atau model Factorization Machine/Neural Collaborative Filtering) yang dapat memberikan rekomendasi yang lebih personal.
*   **Tuning Model Deep Learning:** Arsitektur model deep learning dapat dieksplorasi lebih lanjut (jumlah layer, jumlah unit, aktivasi, regularisasi, learning rate). Dimensi embedding juga dapat di-tune.
*   **Feature Engineering Lanjutan:** Mengekstrak fitur yang lebih mendalam dari deskripsi teks (`About the game`) menggunakan teknik NLP lanjutan (misalnya, Word Embeddings seperti Word2Vec, GloVe, atau model bahasa seperti BERT) dapat meningkatkan kualitas representasi konten.
*   **Evaluasi yang Lebih Robust:** Jika memungkinkan, lakukan A/B testing atau pengumpulan feedback pengguna untuk mengevaluasi kualitas rekomendasi secara langsung.
*   **Scalability:** Untuk dataset yang lebih besar, pertimbangkan implementasi yang lebih skalabel untuk perhitungan kemiripan (misalnya, menggunakan Approximate Nearest Neighbors libraries seperti Annoy atau Faiss).
