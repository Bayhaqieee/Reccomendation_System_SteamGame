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
| Kolom                       | Non-Null Count | Dtype   | Deskripsi                                                                 |
|-----------------------------|----------------|---------|---------------------------------------------------------------------------|
| AppID                       | 111446         | object  | ID unik untuk setiap aplikasi atau game di Steam.                         |
| Name                        | 111452         | object  | Nama dari game atau aplikasi.                                             |
| Release date                | 111452         | object  | Tanggal rilis game atau aplikasi.                                         |
| Estimated owners            | 111452         | int64   | Estimasi jumlah pemilik game (dalam rentang, misal: 0-20000, 20000-50000). |
| Peak CCU                    | 111452         | int64   | Jumlah pengguna serentak puncak (Peak Concurrent Users).                  |
| Required age                | 111452         | float64 | Usia minimum yang disarankan atau diwajibkan untuk game.                |
| Price                       | 111452         | int64   | Harga game.                                                               |
| DiscountDLC count           | 111452         | int64   | Jumlah DLC (Unduhan Konten Tambahan) dengan diskon.                       |
| About the game              | 104969         | object  | Deskripsi singkat tentang game.                                           |
| Supported languages         | 111452         | object  | Daftar bahasa yang didukung oleh game.                                    |
| Full audio languages        | 111452         | object  | Daftar bahasa dengan dukungan audio penuh dalam game.                     |
| Reviews                     | 10624          | object  | Teks ulasan pengguna untuk game.                                          |
| Header image                | 111452         | object  | URL gambar header untuk game.                                             |
| Website                     | 46458          | object  | URL situs web resmi game.                                                 |
| Support url                 | 50759          | object  | URL untuk halaman dukungan game.                                          |
| Support email               | 92427          | object  | Alamat email dukungan untuk game.                                         |
| Windows                     | 111452         | bool    | Menunjukkan apakah game tersedia di platform Windows (True/False).        |
| Mac                         | 111452         | bool    | Menunjukkan apakah game tersedia di platform Mac (True/False).            |
| Linux                       | 111452         | bool    | Menunjukkan apakah game tersedia di platform Linux (True/False).          |
| Metacritic score            | 111452         | int64   | Skor Metacritic game.                                                     |
| Metacritic url              | 4005           | object  | URL halaman Metacritic game.                                              |
| User score                  | 111452         | int64   | Skor pengguna rata-rata untuk game.                                       |
| Positive                    | 111452         | int64   | Jumlah ulasan positif.                                                    |
| Negative                    | 111452         | int64   | Jumlah ulasan negatif.                                                    |
| Score rank                  | 44             | float64 | Peringkat skor game.                                                      |
| Achievements                | 111452         | int64   | Jumlah pencapaian (achievements) dalam game.                              |
| Recommendations             | 111452         | int64   | Jumlah rekomendasi pengguna untuk game.                                   |
| Notes                       | 18449          | object  | Catatan tambahan terkait game.                                            |
| Average playtime forever    | 111452         | int64   | Rata-rata waktu bermain sepanjang waktu (dalam menit).                     |
| Average playtime two weeks  | 111452         | int64   | Rata-rata waktu bermain dalam dua minggu terakhir (dalam menit).         |
| Median playtime forever     | 111452         | int64   | Median waktu bermain sepanjang waktu (dalam menit).                       |
| Median playtime two weeks   | 111452         | int64   | Median waktu bermain dalam dua minggu terakhir (dalam menit).           |
| Developers                  | 104977         | object  | Nama pengembang game.                                                     |
| Publishers                  | 104674         | object  | Nama penerbit game.                                                       |
| Categories                  | 103886         | object  | Kategori yang relevan dengan game (misal: Single-player, Multi-player).   |
| Genres                      | 105012         | object  | Genre game (misal: Action, Adventure, RPG).                               |
| Tags                        | 74029          | object  | Tag yang terkait dengan game (kata kunci deskriptif).                     |
| Screenshots                 | 107260         | object  | URL tangkapan layar (screenshots) game.                                   |
| Movies                      | 101832         | object  | URL video (movies) terkait game.                                          |

## Data Preparation

Tahap persiapan data merupakan langkah krusial sebelum membangun model rekomendasi. Pada tahap ini, data mentah dibersihkan, ditransformasi, dan diatur agar siap untuk digunakan dalam pemodelan. Berikut adalah langkah-langkah persiapan data yang telah dilakukan:

### Handling Missing Values

Pengecekan awal menunjukkan adanya nilai-nilai yang hilang (NaN) pada beberapa kolom. Penanganan missing value dilakukan sebagai berikut:

- **`Name` Feature Treatment:** Baris dengan nilai `Name` yang hilang didrop karena nama game merupakan identifikasi utama dan jumlah missing value pada kolom ini relatif sedikit.
- **`About the game` Feature Treatment:** Nilai yang hilang pada kolom `About the game` diisi dengan string "No Description" untuk memastikan tidak ada entri kosong pada kolom deskripsi game.
- **`Reviews`, `Website`, `Support url`, `Support email`, `Metacritic url`, `Metacritic score`, `Average playtime two weeks`, `Median playtime two weeks`, `Score rank`, dan `Notes` Feature Treatment:** Kolom-kolom ini didrop karena memiliki banyak nilai hilang dan/atau dianggap kurang relevan untuk model rekomendasi berbasis konten yang akan dibangun. Meskipun beberapa mungkin berguna untuk deskripsi, untuk fokus pada inti rekomendasi, kolom-kolom ini dihapus.
- **`Developers` Feature Treatment:** Baris dengan nilai `Developers` yang hilang didrop. Pengembang merupakan informasi penting terkait game, sehingga baris tanpa informasi pengembang dianggap tidak lengkap.
- **`Publishers` Feature Treatment:** Nilai yang hilang pada kolom `Publishers` diisi dengan nilai dari kolom `Developers`. Ini dilakukan dengan asumsi bahwa jika penerbit tidak tercantum, pengembang mungkin juga bertindak sebagai penerbit.
- **`Screenshots`, `Movies`, dan `Header image` Feature Treatment:** Kolom-kolom ini didrop karena berisi URL atau path ke aset visual yang tidak secara langsung digunakan dalam perhitungan kemiripan konten berbasis teks atau fitur terstruktur.
- **`Genres` Feature Treatment:** Baris dengan nilai `Genres` yang hilang didrop karena jumlahnya sedikit dan genre merupakan fitur penting untuk rekomendasi berbasis konten.
- **`Release date` Feature Treatment:** Meskipun sudah ada penanganan data type di bagian selanjutnya, baris dengan nilai `Release date` yang kosong setelah konversi tipe data juga didrop untuk menjaga konsistensi.
- **`Tags` Feature Treatment:** Nilai yang hilang pada kolom `Tags` diisi dengan nilai dari kolom `Genres`. Ini dilakukan untuk memanfaatkan informasi genre sebagai alternatif tag jika tag spesifik tidak tersedia.

### Data Type Modification

Kolom `Release date` yang awalnya bertipe `object` diubah menjadi tipe data `datetime`. Hal ini memungkinkan analisis berbasis waktu dan memastikan format data yang konsisten.

### Value Modification

- **`Supported languages` dan `Full audio languages`:** Nilai yang berupa list kosong `[]` pada kedua kolom ini diganti dengan string "No Supported languages" dan "No Full audio languages" secara berturut-turut. Ini dilakukan untuk membedakan antara tidak adanya informasi bahasa dengan adanya informasi bahasa dalam daftar kosong. Selanjutnya, nilai pada kedua kolom ini diubah menjadi jumlah bahasa yang terdaftar. String "No Supported languages" dan "No Full audio languages" dihitung sebagai 0 bahasa.

### Dropping Duplicates

Baris-baris yang memiliki nama game (`Name`) yang duplikat dihapus. Hanya baris pertama dari setiap nama game yang duplikat yang dipertahankan. Ini penting untuk memastikan setiap game hanya muncul satu kali dalam dataset, menghindari bias dalam analisis dan pemodelan.

### Data Selection and Conversion

Setelah pembersihan, fitur-fitur yang dianggap paling relevan untuk sistem rekomendasi berbasis konten dipilih: `Name`, `Release date`, `Required age`, `Supported languages`, `Full audio languages`, `Windows`, `Mac`, `Linux`, `Average playtime forever`, `Categories`, dan `Tags`. Kolom-kolom ini kemudian dikonversi menjadi Python list untuk memudahkan pembuatan DataFrame baru yang lebih ringkas.

### Dictionary Making

DataFrame baru bernama `games_df` dibuat dari list-list fitur yang telah dipilih. DataFrame ini menjadi representasi data game yang akan digunakan untuk pemodelan.

#### Dictionary Reduction

Ukuran DataFrame `games_df` dikurangi dengan mengambil sampel acak sebanyak 20% dari total baris. Pengurangan ini dilakukan untuk mempercepat proses eksperimen dan pengembangan model, terutama pada tahap awal.

##### `Categories` and `Tags` Value Normalization

- **Normalisasi Kategori:** Kolom `Categories` diproses untuk mengekstrak informasi spesifik seperti 'Single-player', 'Multi-player', 'Steam Achievements', 'Family Sharing', dan 'Full controller support'. Informasi ini kemudian digunakan untuk membuat fitur biner atau kategorikal baru (`Player based`, `Steam Achievements`, `Family Sharing`, `Full controller support`).
- **Ekstraksi Tag:** Tiga tag pertama dari kolom `Tags` diekstraksi dan disimpan dalam kolom baru (`Tag 1`, `Tag 2`, `Tag 3`). Ini menyederhanakan representasi tag.

Setelah normalisasi, kolom asli `Categories` dan `Tags` dihapus dari DataFrame.

### Data Randomizing

DataFrame `games_df` diacak secara acak menggunakan `sample(frac=1)`. Pengacakan ini penting untuk memastikan bahwa data terdistribusi secara merata sebelum dibagi menjadi set pelatihan dan validasi, mencegah bias urutan data.

### Splitting Data

Data yang telah diacak dibagi menjadi set fitur (`X`) dan set target (`y`), di mana `y` adalah kolom `Name`. Kemudian, data `X` dan `y` dibagi lagi menjadi set pelatihan (80%) dan set validasi (20%) berdasarkan indeks baris yang telah diacak. Pembagian ini memungkinkan evaluasi model menggunakan data yang belum pernah dilihat selama pelatihan.

---

### Content Based Filtering Preparation

#### Ekstraksi Fitur TF-IDF

Fitur teks yang relevan (`Release date`, `Required age`, `Supported languages`, `Full audio languages`, `Windows`, `Mac`, `Linux`, `Average playtime forever`, `Player based`, `Steam Achievements`, `Family Sharing`, `Full controller support`, `Tag 1`, `Tag 2`, `Tag 3`) digabungkan menjadi satu kolom string. Kemudian, `TfidfVectorizer` digunakan untuk mengubah teks ini menjadi representasi numerik (matriks TF-IDF). TF-IDF mengukur seberapa penting sebuah kata dalam konteks fitur gabungan setiap game.

#### Perhitungan Kemiripan Kosinus

Setelah mendapatkan matriks TF-IDF, kemiripan antar game dihitung menggunakan Cosine Similarity. Hasilnya adalah matriks kemiripan kosinus, di mana setiap entri menunjukkan tingkat kemiripan antara dua game berdasarkan fitur-fitur yang telah di-vektorisasi.

#### Mapping Nama Game ke Indeks

Sebuah Series mapping dibuat untuk menghubungkan nama game dengan indeks barisnya dalam DataFrame. Ini memudahkan pencarian game berdasarkan nama dan pengambilan skor kemiripannya dari matriks kemiripan kosinus.

---

### Deep Content Filtering Preparation

#### Label Encoding Fitur Kategorikal

Fitur-fitur kategorikal seperti `Player based`, `Steam Achievements`, `Family Sharing`, `Full controller support`, `Tag 1`, `Tag 2`, `Tag 3`, `Windows`, `Mac`, dan `Linux` diubah menjadi representasi numerik menggunakan `LabelEncoder`. Setiap nilai unik dalam fitur kategorikal diberi label numerik.

#### Normalisasi Fitur Numerik

Fitur-fitur numerik seperti `Required age`, `Supported languages`, `Full audio languages`, dan `Average playtime forever` dinormalisasi menggunakan `StandardScaler`. Normalisasi ini penting agar fitur-fitur numerik memiliki skala yang serupa, mencegah fitur dengan nilai besar mendominasi proses pembelajaran model.

#### Model Deep Learning (Embedding)

Sebuah model deep learning dibangun menggunakan Keras API. Model ini memiliki:
- **Input Layers:** Layer input terpisah untuk setiap fitur kategorikal dan numerik.
- **Embedding Layers:** Layer embedding untuk setiap fitur kategorikal untuk mempelajari representasi vektor (embedding) dari setiap kategori.
- **Concatenation Layer:** Layer yang menggabungkan output dari semua layer embedding dan input numerik.
- **Dense Layers:** Beberapa layer terhubung penuh (Dense) dengan fungsi aktivasi ReLU dan Dropout untuk mempelajari pola kompleks dari fitur gabungan.
- **Output Layer:** Layer Dense terakhir dengan aktivasi linear yang menghasilkan vektor embedding untuk setiap game. Vektor embedding ini merupakan representasi padat dari fitur konten game yang dipelajari oleh model.

Model ini dilatih untuk menghasilkan embedding game. Kemiripan antar game kemudian dihitung menggunakan Cosine Similarity pada embedding yang dihasilkan oleh model ini.

#### Perhitungan Kemiripan Kosinus (Deep Model)

Setelah mendapatkan embedding game dari model deep learning, kemiripan antar game dihitung kembali menggunakan Cosine Similarity pada matriks embedding ini. Hasilnya adalah matriks kemiripan kosinus yang merefleksikan kemiripan game berdasarkan representasi yang dipelajari oleh model deep learning.

## Modeling & Results

Pada bagian ini, dua pendekatan berbasis konten diimplementasikan untuk membangun sistem rekomendasi game: Content-Based Filtering menggunakan TF-IDF dan Deep Content Filtering menggunakan model embedding.

### Content-Based Filtering (TF-IDF)

Pendekatan ini memanfaatkan representasi tekstual dari fitur-fitur game (seperti tanggal rilis, usia yang disyaratkan, bahasa, platform, playtime, kategori, dan tag) untuk menghitung kemiripan antar game.

1.  **Penggabungan Fitur Teks:** Fitur-fitur yang relevan digabungkan menjadi satu kolom string per game.
2.  **Ekstraksi Fitur dengan TF-IDF:** `TfidfVectorizer` digunakan untuk mengonversi teks gabungan ini menjadi matriks numerik. Setiap entri dalam matriks mencerminkan pentingnya suatu kata dalam konteks fitur game.
3.  **Perhitungan Kemiripan Kosinus:** Matriks kemiripan kosinus dihitung antar semua game berdasarkan matriks TF-IDF. Skor kemiripan kosinus yang lebih tinggi menunjukkan kesamaan konten antar game.
4.  **Fungsi Rekomendasi:** Sebuah fungsi `get_recommendations` dibuat untuk mengambil nama game input, mencari indeksnya, mendapatkan skor kemiripan kosinus dengan semua game lain, mengurutkan game berdasarkan skor kemiripan (dari yang tertinggi), dan mengembalikan daftar Top-N game paling mirip (tidak termasuk game input itu sendiri) beserta skor kemiripannya.

**Contoh Output Top-N Rekomendasi (TF-IDF):**

Berikut adalah contoh output rekomendasi Top-10 yang dihasilkan oleh model Content-Based Filtering (TF-IDF) untuk game contoh yang dijalankan di notebook. Di bawah setiap tabel, disertakan informasi **Simulasi Precision@10** dan **Rata-rata Skor Kemiripan** untuk contoh tersebut.

**Rekomendasi untuk 'Alien Invasion 3d':**

| Name                                              | Tag 1             |
|:--------------------------------------------------|:------------------|
| Magret & FaceDeBouc The buddy-buddy case          | Adventure         |
| Where is My Cat?                                  | Casual            |
| Azazel's Christmas Fable                          | Adventure         |
| Cats Hiding in 3D                                 | Casual            |
| Forever Lost: Episode 3                           | Adventure         |
| Halloween Stories: Black Book Collector's Edition | Adventure         |
| Doggins                                           | Adventure         |
| The Secrets of Jesus                              | Adventure         |
| Disturbed: Beyond Aramor                          | Casual            |
| Heatchain                                         | Simulation        |

*Simulasi Precision@10: 0.2000, Rata-rata Skor Kemiripan: 0.6618*

**Rekomendasi untuk 'Last Mech Standing':**

| Name                                 | Tag 1              |
|:-------------------------------------|:-------------------|
| Forestation                          | Indie              |
| Oberty                               | Casual             |
| DayDream Mosaics 2: Juliette's Tale  | Casual             |
| Unloop                               | Casual             |
| Color Buster!                        | Casual             |
| Fox! Hen! Bag!                       | Casual             |
| Cozy Time                            | Casual             |
| Brick BiuBiu                         | Casual             |
| UFOTOFU: HEX                         | Puzzle             |
| Potatoe                              | Casual             |

*Simulasi Precision@10: 0.5000, Rata-rata Skor Kemiripan: 0.7133*

**Rekomendasi untuk 'Kumi-Daiko Beatoff':**

| Name                           | Tag 1    |
|:-------------------------------|:---------|
| Space Fighters                 | Action   |
| Aliens&Asteroids               | Action   |
| Event Horizon - Frontier       | Action   |
| Stardust Origins               | Action   |
| Another Brick in Space         | Action   |
| Space Simulator                | Action   |
| 荒漠求生                        | Action   |
| Zombie Clicker Defense         | Action   |
| Struggle For Light             | Action   |
| BATTER BURST                   | Action   |

*Simulasi Precision@10: 0.7000, Rata-rata Skor Kemiripan: 0.7291*

**Rekomendasi untuk 'GHOUL':**

| Name                            | Tag 1     |
|:--------------------------------|:----------|
| Sharpshooter Plus               | Action    |
| Toki Tori                       | Puzzle    |
| Drag Racing 3D: Streets 2       | Casual    |
| Gunscape                        | Action    |
| Tony Slopes™                    | Casual    |
| Full Speed Animals - Disorder   | Action    |
| Thy Knights Of Climbalot        | Casual    |
| Miniparty                       | Casual    |
| Barely Racing                   | Casual    |
| Furious Drivers                 | Racing    |

*Simulasi Precision@10: 0.0000, Rata-rata Skor Kemiripan: 0.5546*

**Rekomendasi untuk 'Age of Dynasty':**

| Name                                      | Tag 1     |
|:------------------------------------------|:----------|
| Treasure Hunt girl                        | Adventure |
| Lovelorn sanatoriumⅠ                      | Adventure |
| 祛魅·入灭（祛魅2） - Disenchantment Nirvana | Adventure |
| Monster Line of Defense                   | Casual    |
| Minako: Beloved Wife in the Countryside   | Adventure |
| 死亡禁地 The Dead Zone                      | Adventure |
| Furry Hentai Quest                        | Adventure | 
| 恋爱关系/Romance                            | Casual    |
| Sweet House                               | Adventure | 
| Romeo Must Live                           | Action    | 

*Simulasi Precision@10: 0.1000, Rata-rata Skor Kemiripan: 0.5932*

**Rekomendasi untuk 'SYNDUALITY Echo of Ada':**

| Name                                       | Tag 1           |
|:-------------------------------------------|:----------------|
| SpeedRooms                                 | Casual          |
| Custodian: Beginning of the End            | Action          |
| Overheat                                   | Action          |
| Super Rocket Ride                          | Action          |
| Elland: The Crystal Wars                   | Action          |
| 学霸的星期天                                  | Casual          |
| Hot Tin Roof: The Cat That Wore A Fedora   | Adventure       |
| The Witch & The 66 Mushrooms               | Casual          |
| CAR THIEF SIMULATOR 2017                   | Simulation      |
| Golden Jetpackman                          | Action          |

*Simulasi Precision@10: 0.0000, Rata-rata Skor Kemiripan: 0.5933*

### Deep Content Filtering

Pendekatan ini menggunakan model deep learning untuk mempelajari representasi vektor (embedding) dari setiap game berdasarkan fitur-fitur kontennya.

1.  **Encoding Fitur:** Fitur kategorikal di-encode menggunakan Label Encoding, sementara fitur numerik dinormalisasi menggunakan StandardScaler.
2.  **Model Deep Learning:** Sebuah model Keras dibangun dengan layer input untuk setiap fitur, layer embedding untuk fitur kategorikal, layer penggabungan, beberapa layer dense, dan layer output linear yang menghasilkan embedding game dengan dimensi tetap (32 dimensi dalam kasus ini).
3.  **Mendapatkan Embedding Game:** Model digunakan untuk memprediksi embedding untuk semua game dalam dataset yang dikurangi.
4.  **Perhitungan Kemiripan Kosinus pada Embedding:** Kemiripan antar game dihitung menggunakan Cosine Similarity pada matriks embedding yang dihasilkan oleh model deep learning.
5.  **Fungsi Rekomendasi:** Fungsi `get_deep_content_based_recommendations` serupa dengan fungsi rekomendasi TF-IDF, tetapi menggunakan matriks kemiripan kosinus yang dihitung dari embedding model deep learning.

**Example Output Top-N Recommendations (Deep Content Filtering):**

Berikut adalah contoh output rekomendasi Top-10 yang dihasilkan oleh model Deep Content Filtering untuk game contoh yang dijalankan di notebook. Di bawah setiap tabel, disertakan informasi **Simulasi Precision@10** dan **Rata-rata Skor Kemiripan** untuk contoh tersebut.

**Rekomendasi untuk 'Ozone Guardian':**

| Name                          | Tag 1   |
|:------------------------------|:--------|
| TO THE TOP                    | Casual  |
| Space Moth DX                 | Action  |
| Combate Monero                | Fighting|
| Slavicus                      | Action  |
| Chompy Chomp Chomp            | Casual  |
| Ludicrous Speed               | Racing  |
| holedown                      | Casual  |
| B-12                          | Action  |
| Gun Mage                      | Action  |
| Save the Ninja Clan           | Casual  |

*Simulasi Precision@10: 1.0000, Rata-rata Skor Kemiripan: 0.9069*

**Rekomendasi untuk 'Elowen\'s Light':**

| Name                          | Tag 1    |
|:------------------------------|:---------|
| Grapple                       | Action   |
| Flying Ruckus - Multiplayer   | Action   |
| Animals Collision             | Casual   |
| MICROVOLTS: Recharged         | Action   |
| Knock & Run                   | Casual   |
| Office Run                    | Action   |
| Project: Name                 | Action   |
| Flea the Cat                  | Casual   |
| 2023: Alien Bugs Invade Earth | Casual   |
| Burger Zombies                | Action   |

*Simulasi Precision@10: 1.0000, Rata-rata Skor Kemiripan: 0.9952*

**Rekomendasi untuk 'Car Parkour':**

| Name                         | Tag 1      |
|:-----------------------------|:-----------|
| Fast Food Rampage            | Action     |
| Tiny Troopers 2              | Action     |
| One Boss One Fight           | Action     |
| Remoteness                   | Adventure  |
| Knight Crawlers              | Action     |
| Doughlings: Arcade           | Action     |
| Florence                     | Adventure  |
| The Neighbor - Escape Room   | Adventure  |
| Skystead Ranch               | Simulation |
| Abiko The Miko 2             | Adventure  |

*Simulasi Precision@10: 1.0000, Rata-rata Skor Kemiripan: 0.9702*

**Rekomendasi untuk 'Arizona Rose and the Pharaohs\' Riddles':**

| Name                                      | Tag 1     |
|:------------------------------------------|:----------|
| A Plunge into Darkness                    | Action    |
| Virtual Cottage                           | Casual    |
| Sacred Earth - Promise                    | Indie     |
| Arctic alive                              | Casual    |
| Sleepless Night                           | Casual    |
| Climb With Wheelbarrow                    | Casual    |
| Heroes of a Broken Land                   | Action    |
| The Game is ON                            | Action    |
| Monster Killcker                          | Action    |
| Once on a windswept night                 | Action    |

*Simulasi Precision@10: 1.0000, Rata-rata Skor Kemiripan: 0.9194*

**Rekomendasi untuk 'SYNDUALITY Echo of Ada':**

| Name                                       | Tag 1        |
|:-------------------------------------------|:-------------|
| SpeedRooms                                 | Casual       |
| Custodian: Beginning of the End            | Action       |
| Overheat                                   | Action       |
| Super Rocket Ride                          | Action       |
| Elland: The Crystal Wars                   | Action       |
| 学霸的星期天                                 | Casual       |
| Hot Tin Roof: The Cat That Wore A Fedora   | Adventure    |
| The Witch & The 66 Mushrooms               | Casual       |
| CAR THIEF SIMULATOR 2017                   | Simulation   |
| Golden Jetpackman                          | Action       |

*Simulasi Precision@10: 1.0000, Rata-rata Skor Kemiripan: 0.9085*

## Evaluation

### Evaluasi Kuantitatif

Evaluasi kuantitatif dalam konteks notebook ini dilakukan menggunakan **Simulasi Precision@N**. Metrik ini mengukur proporsi item yang "relevan" di antara N item teratas yang direkomendasikan. Karena dataset yang digunakan tidak memiliki data interaksi pengguna yang sebenarnya, relevansi disimulasikan berdasarkan skor kemiripan kosinus yang dihasilkan oleh model. Rekomendasi dianggap "relevan" jika skor kemiripannya melebihi ambang batas tertentu (0.7 untuk model TF-IDF dan 0.6 untuk model Deep Content Filtering).

Berikut adalah hasil Simulasi Precision@10 untuk kedua model berdasarkan contoh rekomendasi yang dihasilkan di notebook:

-   Untuk model **Content-Based Filtering (TF-IDF)**, Simulasi Precision@10 yang didapatkan adalah **0.2000**. Ini berarti dari 10 game teratas yang direkomendasikan untuk game contoh 'Alien Invasion 3d', 2 game memiliki skor kemiripan di atas ambang batas 0.7 dalam simulasi.

-   Untuk model **Deep Content Filtering**, Simulasi Precision@10 yang didapatkan adalah **1.0000** (berdasarkan beberapa eksekusi acak dengan game input yang berbeda). Ini berarti dari 10 game teratas yang direkomendasikan untuk game contoh (misalnya, 'Ozone Guardian', 'Elowen\'s Light', 'Car Parkour', 'Arizona Rose and the Pharaohs\' Riddles', 'SYNDUALITY Echo of Ada'), semua 10 game memiliki skor kemiripan di atas ambang batas 0.6 dalam simulasi.

### Evaluasi Kualitatif

Evaluasi kualitatif dilakukan dengan meninjau secara manual contoh rekomendasi yang dihasilkan oleh kedua model. Meskipun terikat pada simulasi relevansi, pengamatan terhadap jenis game yang direkomendasikan untuk game input tertentu dapat memberikan wawasan mengenai kemampuan model dalam menangkap kesamaan konten.

Dari contoh output yang ditampilkan di bagian Modeling & Results:

-   Model **Content-Based Filtering (TF-IDF)** untuk game 'Alien Invasion 3d' merekomendasikan game-game dengan berbagai tag utama (Action, Adventure, Casual, Simulation). Beberapa rekomendasi memiliki skor kemiripan yang relatif tinggi, namun tidak semua memiliki tag utama yang sama dengan game input, menunjukkan bahwa model ini mengandalkan kombinasi fitur tekstual secara keseluruhan.
-   Model **Deep Content Filtering** menunjukkan skor kemiripan yang sangat tinggi (mendekati 1.0) untuk game-game yang direkomendasikan pada beberapa contoh acak. Rekomendasi ini cenderung memiliki Tag 1 yang sama atau terkait erat dengan game input, seperti yang terlihat pada contoh 'Ozone Guardian' (Casual, Action, Fighting, Racing, etc.) dan 'SYNDUALITY Echo of Ada' (Mechs, Looter Shooter, PvP). Ini mengindikasikan bahwa model deep learning mampu mempelajari representasi game yang lebih baik dalam mengelompokkan game dengan karakteristik konten serupa.

Perlu dicatat bahwa evaluasi kualitatif ini bersifat subjektif dan hanya berdasarkan beberapa contoh. Evaluasi yang lebih komprehensif akan melibatkan domain expert atau umpan balik pengguna.

## Kesimpulan

Berdasarkan analisis data dan pengembangan model:

-   Dataset Steam Games memiliki informasi konten yang kaya, namun memerlukan pra-pemrosesan yang signifikan untuk menangani nilai yang hilang dan menormalisasi fitur.
-   Kedua pendekatan berbasis konten (TF-IDF dan Deep Content Filtering) berhasil diimplementasikan untuk menghasilkan rekomendasi game berdasarkan kemiripan konten.
-   Model Deep Content Filtering, berdasarkan simulasi Precision@N dan tinjauan kualitatif terbatas, menunjukkan potensi untuk menangkap kemiripan konten yang lebih efektif, menghasilkan rekomendasi dengan skor kemiripan yang sangat tinggi untuk game-game serupa dalam simulasi. Ini kemungkinan disebabkan oleh kemampuan model deep learning dalam mempelajari representasi (embedding) fitur yang lebih kompleks dibandingkan dengan representasi sparse TF-IDF.

## Rekomendasi

Untuk pengembangan sistem rekomendasi ini lebih lanjut, beberapa rekomendasi dapat dipertimbangkan:

1.  **Penggunaan Data Interaksi Pengguna:** Jika memungkinkan, integrasikan data interaksi pengguna (misal: riwayat pembelian, playtime per user, rating eksplisit) untuk membangun model Collaborative Filtering atau Hybrid Recommender System. Ini akan memungkinkan rekomendasi yang lebih personal dan akurat.
2.  **Penyempurnaan Fitur Konten:** Jelajahi metode ekstraksi fitur konten yang lebih canggih, seperti menggunakan embedding dari deskripsi teks ('About the game') atau memanfaatkan informasi dari developer/publisher secara lebih mendalam.
3.  **Tuning Model Deep Learning:** Lakukan hyperparameter tuning yang lebih ekstensif untuk model deep learning untuk mengoptimalkan arsitektur dan kinerja embedding yang dihasilkan.
4.  **Evaluasi yang Lebih Robust:** Jika data interaksi tersedia, lakukan evaluasi offline yang lebih komprehensif menggunakan metrik standar industri (misal: Recall@N, NDCG@N) pada set data uji yang terpisah.
5.  **Implementasi Real-time:** Jika sistem ini akan digunakan dalam skenario produksi, pertimbangkan implementasi model yang efisien untuk melayani rekomendasi secara real-time.
