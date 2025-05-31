# Laporan Proyek Machine Learning - Muhammad Aditya Bayhaqie

## Domain Proyek

Proyek ini berfokus pada pengembangan sistem rekomendasi game menggunakan data dari Steam. Domain proyek ini berada dalam ranah **Sistem Rekomendasi Konten (Content-Based Recommendation Systems)**, yang menganalisis fitur-fitur item (dalam hal ini, game) untuk merekomendasikan item serupa.

## Business Understanding

### Problem Statements

Dalam pasar game digital yang sangat kompetitif dan luas seperti Steam, pengguna seringkali kesulitan menemukan game baru yang sesuai dengan minat dan preferensi mereka di antara puluhan ribu judul yang tersedia. Kurangnya sistem penemuan game yang efektif dapat menyebabkan pengguna melewatkan game yang berpotensi mereka sukai, mengurangi kepuasan pengguna, dan membatasi eksposur game dari pengembang.

### Goals

1.  Membangun sistem rekomendasi game berbasis konten yang dapat memberikan saran game yang relevan kepada pengguna berdasarkan game yang sudah mereka sukai atau minati.
2.  Meningkatkan pengalaman pengguna dengan mempermudah penemuan game baru yang sesuai selera.
3.  Menyediakan alat bantu bagi pengembang game untuk meningkatkan visibilitas game mereka kepada audiens yang tepat.

### Solution Statements

Mengembangkan dua model sistem rekomendasi berbasis konten menggunakan dataset Steam Games:
1.  Model berbasis **TF-IDF dan Cosine Similarity** untuk menghitung kemiripan antar game berdasarkan fitur teks.
2.  Model berbasis **Deep Learning** (embedding) untuk mempelajari representasi fitur game yang lebih kaya dan menghitung kemiripan antar game dalam ruang embedding.
Kedua model ini akan diproses dan dibangun di lingkungan Google Colab.

## Data Understanding

### URL/Tautan Sumber Data

Dataset yang digunakan bersumber dari Kaggle: [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data?select=games.csv)

### Jumlah Baris dan Kolom

Dataset asli memiliki:
-   Jumlah Baris: 111452
-   Jumlah Kolom: 37

Setelah proses pembersihan dan pemilihan fitur:
-   Jumlah Baris (pada subset yang digunakan untuk modelling): sekitar 20% dari total baris unik (sekitar 22000an baris)
-   Jumlah Kolom (pada DataFrame `games_df` setelah preprocessing): 11 (sebelum penambahan fitur hasil normalisasi kategori/tag) / sekitar 16 (setelah penambahan fitur hasil normalisasi kategori/tag)

### Kondisi Data

-   Terdapat banyak missing values pada beberapa kolom seperti `Score rank`, `Metacritic url`, `Reviews`, `Website`, `Support url`, `Support email`, `Metacritic score`, `Average playtime two weeks`, `Median playtime two weeks`, `Screenshots`, `Movies`, `Header image`, `Developers`, `Publishers`, `Categories`, `Tags`, dan `Release date`.
-   Terdapat data duplikat berdasarkan nama game.
-   Beberapa kolom mengandung data non-standar atau perlu diekstraksi informasinya lebih lanjut (misalnya `Categories` dan `Tags` dalam format string terpisah koma, `Supported languages` dan `Full audio languages` dengan nilai '[]').

### Fitur dalam Dataset:

Dataset asli mengandung 37 fitur yang meliputi informasi tentang game seperti:
-   `AppID`: ID unik game.
-   `Name`: Nama game.
-   `Release date`: Tanggal rilis.
-   `Estimated owners`: Estimasi jumlah pemilik game.
-   `Peak CCU`: Jumlah pemain serentak tertinggi.
-   `Required age`: Batasan usia.
-   `Price`: Harga game.
-   `DLC count`: Jumlah DLC.
-   `About the game`: Deskripsi singkat game.
-   `Supported languages`: Bahasa yang didukung.
-   `Full audio languages`: Bahasa audio penuh yang didukung.
-   `Reviews`: Jumlah review.
-   `Website`: URL website game.
-   `Support url`: URL dukungan.
-   `Support email`: Email dukungan.
-   `Metacritic url`: URL Metacritic.
-   `Metacritic score`: Skor Metacritic.
-   `Coming soon`: Status rilis "Coming Soon".
-   `Type`: Tipe entri (game, dlc, dll).
-   `Categories`: Kategori game (single-player, multi-player, dll).
-   `Genres`: Genre game.
-   `Developers`: Pengembang.
-   `Publishers`: Penerbit.
-   `Tags`: Tag game.
-   `Screenshots`: URL screenshot.
-   `Movies`: URL video/trailer.
-   `Header image`: URL gambar header.
-   `Recommendation count`: Jumlah rekomendasi pengguna.
-   `Average playtime forever`: Rata-rata waktu bermain seumur hidup.
-   `Average playtime two weeks`: Rata-rata waktu bermain dua minggu terakhir.
-   `Median playtime forever`: Median waktu bermain seumur hidup.
-   `Median playtime two weeks`: Median waktu bermain dua minggu terakhir.
-   `Score rank`: Ranking skor.
-   `Rating`: Rating (Overall, Very Positive, dll).
-   `Positive reviews`: Jumlah review positif.
-   `Negative reviews`: Jumlah review negatif.
-   `Achievements`: Jumlah achievements.
-   `Background`: URL gambar latar belakang.
-   `Linux`, `Mac`, `Windows`: Dukungan platform (Boolean).
-   `DLC`, `Supported languages - comma separated`, `Genres - comma separated`, `Tags - comma separated`: Duplikat atau variasi fitur teks.
-   `Release date - dd-mm-yyyy`: Tanggal rilis format lain.
-   `Notes`: Catatan tambahan.
  
Fitur yang dipilih untuk pemodelan adalah: `Name`, `Release date`, `Required age`, `Supported languages`, `Full audio languages`, `Windows`, `Mac`, `Linux`, `Average playtime forever`, `Categories`, dan `Tags`.

## Data Preparation

Tahapan data preparation meliputi:
-   Penanganan Missing Values: Drop baris pada fitur `Name`, `Developers`, `Categories`, `Genres`, `Release date`. Isi nilai kosong pada `About the game` dengan 'No Description', `Publishers` dengan nilai `Developers`, dan `Tags` dengan nilai `Genres`. Drop kolom yang tidak relevan atau memiliki banyak missing value (`Reviews`, `Website`, `Support url`, `Support email`, `Metacritic url`, `Metacritic score`, `Score rank`, `Notes`, `Average playtime two weeks`, `Median playtime two weeks`, `Screenshots`, `Movies`, `Header image`).
-   Modifikasi Tipe Data: Mengubah tipe data `Release date` menjadi datetime.
-   Modifikasi Nilai: Mengganti nilai '[]' pada `Supported languages` dan `Full audio languages` dengan 'No Supported languages' / 'No Full audio languages' dan kemudian mengubahnya menjadi hitungan jumlah bahasa.
-   Dropping Duplicates: Menghapus baris duplikat berdasarkan nama game.
-   Data Selection and Conversion: Memilih fitur-fitur relevan dan mengubahnya menjadi list.
-   Dictionary Making & Reduction: Membuat DataFrame baru dengan fitur terpilih dan mengurangi ukurannya menjadi 20% dari total data unik untuk mempercepat eksperimen.
-   Value Normalization: Mengekstraksi dan menormalisasi informasi dari kolom `Categories` dan `Tags` menjadi fitur-fitur biner/kategorikal baru (`Player based`, `Steam Achievements`, `Family Sharing`, `Full controller support`, `Tag 1`, `Tag 2`, `Tag 3`).
-   Data Randomizing: Mengacak dataset dan membaginya menjadi set pelatihan dan validasi (80:20).
-   Encoding dan Scaling: Melakukan Label Encoding pada fitur kategorikal dan Standard Scaling pada fitur numerik untuk model Deep Learning.

## Modeling

### Model yang digunakan:

1.  **Content-Based Filtering (Tradisional):**
    *   Menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengubah fitur teks gabungan menjadi vektor numerik.
    *   Menggunakan **Cosine Similarity** untuk mengukur kemiripan antar vektor TF-IDF game.
2.  **Deep Content Filtering:**
    *   Menggunakan arsitektur **Jaringan Saraf Tiruan** dengan input layers untuk fitur kategorikal dan numerik, embedding layers untuk fitur kategorikal, dense layers, dan output layer yang menghasilkan embedding vektor game.
    *   Menggunakan **Cosine Similarity** untuk mengukur kemiripan antar embedding vektor game.

## Evaluation

### Hasil Evaluasi Model:

Karena dataset tidak memiliki data interaksi pengguna (rating, playtime per user) yang umum digunakan untuk evaluasi sistem rekomendasi, evaluasi dilakukan secara kualitatif berdasarkan hasil rekomendasi yang diberikan untuk game sampel.

-   **Content-Based Filtering (Tradisional):** Memberikan rekomendasi game yang secara intuitif mirip dengan game masukan berdasarkan fitur-fitur konten yang dipertimbangkan (kategori, tag, dll.). Hasil rekomendasi tampak relevan.
-   **Deep Content Filtering:** Juga memberikan rekomendasi game yang mirip. Model ini berpotensi menangkap pola kemiripan yang lebih halus karena menggunakan representasi embedding yang dipelajari dari berbagai fitur.

Tidak ada metrik evaluasi kuantitatif standar (seperti Precision@K, Recall@K, RMSE) yang dapat dihitung secara langsung tanpa data interaksi pengguna.

## Kesimpulan

Proyek ini berhasil membangun sistem rekomendasi game berbasis konten menggunakan dua pendekatan yang berbeda. Kedua model mampu memberikan saran game yang relevan berdasarkan fitur-fitur game itu sendiri. Model tradisional berbasis TF-IDF efektif dan mudah diinterpretasikan, sementara model Deep Learning menawarkan potensi untuk representasi fitur yang lebih kompleks. Kedua sistem ini dapat diimplementasikan untuk membantu pengguna menemukan game baru di platform distribusi digital.

## Rekomendasi

1.  **Validasi dengan Data Interaksi Pengguna:** Jika memungkinkan, gabungkan dataset ini dengan data interaksi pengguna (misalnya, playtime per user, rating) untuk membangun sistem rekomendasi kolaboratif (Collaborative Filtering) atau model hybrid. Data interaksi juga akan memungkinkan evaluasi kuantitatif yang lebih akurat.
2.  **Tuning Hyperparameter Model Deep Learning:** Lakukan tuning hyperparameter (misalnya, dimensi embedding, jumlah dan ukuran dense layers, dropout rate) pada model Deep Learning untuk meningkatkan kualitas representasi embedding.
3.  **Eksplorasi Fitur Lain:** Pertimbangkan untuk memasukkan fitur lain yang mungkin relevan, seperti deskripsi game yang lebih panjang, ulasan pengguna, atau informasi pengembang/penerbit, ke dalam proses pemodelan konten.
4.  **Penerapan Model pada Dataset Penuh:** Setelah eksperimen berhasil pada subset data, terapkan proses preprocessing dan pemodelan pada dataset Steam Games yang lengkap untuk mendapatkan cakupan rekomendasi yang lebih luas.
5.  **Implementasi dan Pengujian A/B:** Implementasikan sistem rekomendasi ini di lingkungan produksi dan lakukan pengujian A/B untuk membandingkan performanya secara langsung dengan metode rekomendasi yang ada (jika ada) dalam hal metrik bisnis seperti tingkat klik (CTR) atau tingkat konversi (CVR) ke pembelian game.
