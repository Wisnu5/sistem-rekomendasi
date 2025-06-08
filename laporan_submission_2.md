# Laporan Proyek Machine Learning Terapan 2 - Sistem Rekomendasi Tempat Wisata

**Wisnu Al Hussaeni - MC001D5Y1239**

## Project Overview
Sektor pariwisata Indonesia menyimpan potensi ekonomi yang besar, namun banyak wisatawan kesulitan menemukan destinasi yang sesuai dengan minat dan kebutuhan mereka karena informasi yang tersebar dan tidak personal. Sistem rekomendasi berbasis machine learning menjadi solusi efektif untuk menyaring informasi dan menyarankan destinasi yang relevan dan menarik bagi setiap individu. Sistem ini tidak hanya meningkatkan kepuasan wisatawan, namun juga membantu destinasi wisata yang kurang dikenal mendapatkan eksposur yang lebih luas.

## Mengapa ini penting?
- **Personalisasi**: Memberikan rekomendasi yang sesuai dengan preferensi individu meningkatkan pengalaman pengguna.
- **Engagement**: Meningkatkan interaksi pengguna dengan platform wisata.
- **Promosi destinasi**: Membantu destinasi wisata yang kurang populer mendapatkan eksposur lebih luas.
- **Efisiensi**: Mengurangi waktu yang diperlukan wisatawan untuk menemukan destinasi yang cocok.

## Business Understanding
### Problem Statements
1. Bagaimana cara menyarankan destinasi wisata yang relevan berdasarkan deskripsi atau nama tempat wisata?
2. Bagaimana cara mengidentifikasi preferensi pengguna dari riwayat interaksi untuk memberikan rekomendasi personal?
3. Bagaimana cara membangun sistem rekomendasi *top-N* yang efisien dan akurat dalam memberikan hasil yang relevan?

### Goals
1. Mengembangkan model *content-based filtering* menggunakan TF-IDF dan *cosine similarity* untuk merekomendas Caterikan destinasi berdasarkan kemiripan deskripsi atau kategori.
2. Membangun model *collaborative filtering* berbasis *neural network embedding* untuk merekomendasikan destinasi berdasarkan riwayat interaksi pengguna.
3. Menyusun sistem rekomendasi *top-N* yang akurat, efisien, dan dapat digunakan dalam skenario nyata.

### Solution Approach
1. Memanfaatkan deskripsi destinasi wisata.
2. Menggunakan user-item rating matrix.
3. Menggunakan TF-IDF untuk representasi vektor, dan cosine similarity untuk mengukur kemiripan antar destinasi.
4. Membangun neural network dengan embedding layer untuk mempelajari representasi pengguna dan destinasi.

## Data Understanding
Dataset digunakan dari [Kaggle](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination/data)

### 1. Dataset `user.csv`
- **Jumlah Data**: 300 baris, 3 kolom
- **Deskripsi Fitur**:
  - `User_Id` (integer): Identifikasi unik untuk setiap pengguna. Nilai berkisar dari 1 hingga 300, tanpa duplikasi.
  - `Location` (string): Lokasi pengguna, berupa kota dan provinsi (contoh: "Semarang, Jawa Tengah"). Terdapat 28 lokasi unik.
  - `Age` (integer): Usia pengguna, berkisar antara 18 hingga 40 tahun.
- **Kondisi Data**:
  - Tidak ada nilai yang hilang (*missing values*).
  - Tidak ada duplikasi data.
  - Tidak ada outlier signifikan pada kolom `Age`, dengan distribusi usia yang cukup merata (lihat distribusi usia pada EDA).
- **Penggunaan**: Dataset ini digunakan untuk memahami profil pengguna dan mendukung analisis preferensi berdasarkan lokasi dan usia.

### 2. Dataset `tourism_rating.csv`
- **Jumlah Data**: 10.000 baris, 3 kolom
- **Deskripsi Fitur**:
  - `User_Id` (integer): ID pengguna yang memberikan rating, merujuk ke `user.csv`.
  - `Place_Id` (integer): ID destinasi wisata, merujuk ke `tourism_with_id.csv`.
  - `Place_Ratings` (integer): Rating yang diberikan pengguna untuk destinasi, berkisar dari 1 hingga 5.
- **Kondisi Data**:
  - Tidak ada nilai yang hilang.
  - Tidak ada duplikasi data.
  - Distribusi rating menunjukkan variasi preferensi pengguna, dengan rating tertinggi (5) dan terendah (1) tersebar secara wajar.
- **Penggunaan**: Dataset ini digunakan untuk membangun model *collaborative filtering* berdasarkan interaksi pengguna dengan destinasi wisata.

### 3. Dataset `tourism_with_id.csv`
- **Jumlah Data**: 437 baris, 12 kolom
- **Deskripsi Fitur**:
  - `Place_Id` (integer): Identifikasi unik untuk destinasi wisata.
  - `Place_Name` (string): Nama destinasi wisata (contoh: "Monumen Yogya Kembali").
  - `Description` (string): Deskripsi destinasi wisata.
  - `Category` (string): Kategori destinasi, seperti "Budaya", "Bahari", "Taman Hiburan", dll.
  - `City` (string): Kota tempat destinasi berada.
  - `Price` (integer): Harga tiket masuk destinasi.
  - `Rating` (float): Rata-rata rating destinasi.
  - `Time_Minutes` (float): Durasi kunjungan rata-rata (dalam menit).
  - `Coordinate` (string): Koordinat geografis destinasi.
  - `Lat` (float): Latitude destinasi.
  - `Long` (float): Longitude destinasi.
  - `Unnamed: 11` (unknown): Kolom tanpa nama, tidak digunakan dalam analisis.
- **Kondisi Data**:
  - Terdapat nilai yang hilang pada kolom `Time_Minutes` (sekitar 50% data).
  - Tidak ada duplikasi data.
  - Kolom `Unnamed: 11` tidak relevan dan diabaikan.
  - Distribusi harga tiket masuk bervariasi antar kota, dengan beberapa destinasi memiliki harga 0 (gratis).
- **Penggunaan**: Dataset ini digunakan untuk *content-based filtering* (menggunakan kolom `Description` dan `Category`) dan untuk memberikan informasi tambahan tentang destinasi dalam rekomendasi.

![image](https://github.com/user-attachments/assets/72c824b6-c5e2-4ac4-acbc-cc3ed80a6a17)

File: CSV berisi informasi user, tour_rate, pack_tour, dan tour_with_id.
Masing-masing CSV memuat dataset terterntu

![image](https://github.com/user-attachments/assets/9adf1a8a-de6d-4d33-869d-20301fbb26fd)
![image](https://github.com/user-attachments/assets/ec8c72a5-5743-4a93-888a-4e5c99a8b205)
![image](https://github.com/user-attachments/assets/7c030ed5-70c4-456c-818c-b76f61d9f856)
![image](https://github.com/user-attachments/assets/14c3a4e1-6b11-46d1-8f2e-d0d787b6e8e3)

Data yang saya gunakan itu data user, tour_rate, dan tour_with_id.
Masing-masing dataset tidak memiliki missing value kecuali pada dataset tour_with_id

![image](https://github.com/user-attachments/assets/4d686926-e7a4-498c-ad40-a1d835b635f5)
![image](https://github.com/user-attachments/assets/c061b125-fc3a-4245-a390-ce37e070a9f0)
![image](https://github.com/user-attachments/assets/2cae2f58-dc42-4797-9ef2-40ecc1f0b0c7)

dengan struktur masing masing datasetnya memiliki ukuran yang berbeda

![image](https://github.com/user-attachments/assets/becfc493-c2bf-40bf-a7c5-490a82e84659)
![image](https://github.com/user-attachments/assets/4608565a-9315-4d3d-bf01-b4113ec28bb4)
![image](https://github.com/user-attachments/assets/09ca0ae8-fe60-432a-a018-f3fb8932238d)
![image](https://github.com/user-attachments/assets/01d42c52-bbf3-460d-9547-07fe1a805b7c)

## Data Preparation
1. Melakukan penggabungan dataset tour_info dengan tour_rate berdasarkan Place_Id

![image](https://github.com/user-attachments/assets/6a5421b3-1e25-43c2-b062-5ce4ad3b1650)

2. Membuat kolom baru dengan menggabungkan 'City' dengan 'Category'

![image](https://github.com/user-attachments/assets/f7925038-2a51-448a-883d-8a131a049925)

3. Mengecek missing value

![image](https://github.com/user-attachments/assets/979d2a8c-18db-44b1-a922-5fbb5eff1bf8)

4. Membuang data duplikat pada variabel preparation untuk keperluan sebelum diubah dalam bentuk list

![image](https://github.com/user-attachments/assets/c362eed2-7842-4d0a-91d6-77cf6b75d12c)

5. Mengonversi data series 'Place_Id', 'Place_Name', 'Category', 'Description', 'City', 'City_Category' dalam bentuk list sehingga memiliki panjang yang sama

![image](https://github.com/user-attachments/assets/7d067d63-2cfd-47d3-a32e-c3c0562d7989)

6. Membuat dictionary untuk data 'place_id', 'place_name', 'place_category', 'place_desc', 'place_city', 'city_category'

![image](https://github.com/user-attachments/assets/e0371778-adf4-4d1a-8a43-12e73f713d20)

7. Membuat Data Sample Model Development Content Based Filtering

![image](https://github.com/user-attachments/assets/a6b1bacb-fa5e-40bb-8b94-d07298e03be9)

8. Insialisasi TfidfVectorizer

![image](https://github.com/user-attachments/assets/4873bb1c-f13a-44a8-abb6-4e97f3aa51ee)

9. Melakukan fit lalu ditransformasikan ke bentuk matrix

![image](https://github.com/user-attachments/assets/7a3f90d1-450e-4a45-b58f-c2316ec0412a)

10. Membuat dataframe untuk melihat tf-idf matrix

![image](https://github.com/user-attachments/assets/930e24f3-1cdc-4cfb-9019-3ac2b17107fe)

11. Menghitung cosine similarity pada matrix tf-idf

![image](https://github.com/user-attachments/assets/e8cbf1ff-37dc-4fb3-b570-660c9bfe91b6)

12. Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa nama place

![image](https://github.com/user-attachments/assets/204e3780-1a77-4480-98bc-ad7dcbee5add)

13. Mengubah PlaceID menjadi list tanpa nilai yang sama untuk kebutuhan modeling content based filtering

![image](https://github.com/user-attachments/assets/f7d5b2f5-f0aa-467a-a4b2-720f6420ee27)

14. Mengubah userID menjadi list tanpa nilai yang sama

![image](https://github.com/user-attachments/assets/fce406a0-f221-4ef0-8de7-78c355332ddb)

14. Mapping User_Id dan Place_Id ke dataframe user dan place

![image](https://github.com/user-attachments/assets/2f066769-a45f-4b37-a70d-80701a5a3737)

15. Melihat jumlah

![image](https://github.com/user-attachments/assets/6f283e79-2ab9-44ac-946a-5c7765644177)

16. Mengacak dataset untuk keperluan spliting data

![image](https://github.com/user-attachments/assets/4b79bc7d-b373-4b5f-930f-adc74e11df46)

17. Membagi menjadi 80% data train dan 20% data test

![image](https://github.com/user-attachments/assets/8481ca77-d982-4ac4-8274-46329fc2a739)

Content-Based Filtering
- **Pendekatan**: Menggunakan TF-IDF untuk mengubah deskripsi destinasi menjadi vektor, lalu menghitung *cosine similarity* untuk menemukan destinasi yang mirip.
- **Proses**:
  - Inisialisasi `TfidfVectorizer` untuk mengonversi kolom `Description` ke matriks TF-IDF.
  - Menghitung *cosine similarity* antar destinasi menggunakan matriks TF-IDF.
  - Membuat fungsi rekomendasi yang menerima nama destinasi (contoh: "Air Mancur Menari") dan mengembalikan *top-N* destinasi dengan kemiripan tertinggi.
- **Hasil**:
  - Untuk destinasi "Air Mancur Menari", rekomendasi yang dihasilkan mencakup destinasi dengan deskripsi serupa, seperti taman hiburan atau destinasi dengan elemen air.
  - Contoh output:
    - Taman Hiburan Rakyat: Taman Hiburan
    - Taman Flora Bratang Surabaya: Taman Hiburan
    - Taman Prestasi: Taman Hiburan
    - Taman Buah Surabaya: Taman Hiburan
    - Taman Barunawati: Taman Hiburan
  - Contoh output lain untuk destinasi "Trans Studio Bandung"
    - Kampung Batu Malakasari: Taman Hiburan
    - Puspa Iptek Sundial: Taman Hiburan
    - Dago Dreampark: Taman Hiburan
    - Chingu Cafe Little Seoul: Taman Hiburan
    - Taman Lansia: Taman Hiburan

13. Memuat data tour_rate untuk Model Development dengan Collaborative Filtering

![image](https://github.com/user-attachments/assets/82f44583-9441-4427-864c-d08827f81bf6)

14. Proses Training

![image](https://github.com/user-attachments/assets/f29cb7e7-61d7-464a-b679-a24a5464e277)

15. Model menggunakan RecommenderNet

![image](https://github.com/user-attachments/assets/5f0d0096-9a6a-4c64-8147-a5a01fcb4ff9)

16. Menggunakan batch_size=8, epoch=100, verbose=1

![image](https://github.com/user-attachments/assets/af62e189-4e76-4511-8883-dc66c6b45b63)

Collaborative Filtering
- **Pendekatan**: Menggunakan *neural network* dengan *embedding layer* untuk mempelajari representasi pengguna dan destinasi berdasarkan data rating.
- **Struktur Model**:
  - **Input**:
    - *Embedding layer* untuk pengguna (dimensi: jumlah pengguna × 50).
    - *Embedding layer* untuk destinasi (dimensi: jumlah destinasi × 50).
  - **Lapisan**:
    - Dua *embedding layer* untuk pengguna dan destinasi, diikuti oleh operasi *dot product* untuk menghitung skor prediksi.
    - *Regularization* (L2) diterapkan untuk mencegah *overfitting*.
    - Fungsi aktivasi sigmoid digunakan untuk menormalkan output ke rentang [0, 1].
  - **Fungsi Loss**: Mean Squared Error (MSE) untuk mengukur perbedaan antara rating prediksi dan aktual.
  - **Optimizer**: Adam dengan *learning rate* default.
  - **Parameter Pelatihan**:
    - *Batch size*: 8
    - *Epochs*: 100
    - *Verbose*: 1 (menampilkan progres pelatihan)
- **Proses Pelatihan**:
  - Dataset `tourism_rating.csv` dibagi menjadi 80% data latih dan 20% data uji.
  - Model dilatih untuk meminimalkan RMSE antara rating prediksi dan aktual.


## Exploratory Data Analysis (EDA)
1. Melihat jumlah kunjungan untuk setiap tempat wisata

![image](https://github.com/user-attachments/assets/8c3057f3-0197-4923-a6d6-5c2c02a4fe2a)
![image](https://github.com/user-attachments/assets/20694e47-a1db-4cc8-9663-7f0b708d00cd)

2. Melihat jumlah kunjungan untuk setiap kategori tempat wisata

![image](https://github.com/user-attachments/assets/7af8df31-9e94-43e2-8589-6e87da7d3a4d)
![image](https://github.com/user-attachments/assets/713d8e0c-27dc-4d76-b822-385a056b63b6)

3. Melihat distribusi usia

![image](https://github.com/user-attachments/assets/b5c8bcc3-3a9c-4556-a692-504045454586)
![image](https://github.com/user-attachments/assets/95c5dce4-5d01-4f1c-b638-b7c4a3dc0ab8)

4. Melihat distribusi harga masuk tiap  kota

![image](https://github.com/user-attachments/assets/83da445d-aa3c-4f42-a131-fba5bde44f36)
![image](https://github.com/user-attachments/assets/a71a83bd-8847-4e2c-b24d-1ea46922f865)

5. Melihat distribusi user

![image](https://github.com/user-attachments/assets/b13011ce-c14c-437e-a227-36bf5d30a39b)
![image](https://github.com/user-attachments/assets/e4a72927-01f5-4a15-b3b5-4ed6cf34da23)

## Mendapatkan Rekomendasi
1. Membuat fungsi rekomendasi

![image](https://github.com/user-attachments/assets/fd66073f-81a8-42b0-865a-6435fe4db09f)

2. Mendapatkan rekomendasi place name yang mirip dengan 'Air Mancur Menari'

![image](https://github.com/user-attachments/assets/00474402-35f5-416c-a8ad-16d9ebd29f2c)

3. Content Based Filtering Evaluasi 10 besar

![image](https://github.com/user-attachments/assets/80b00eb3-8682-4163-9820-7fd960b4f4f1)

Precision tinggi sistem merekomendasikan tepat dalam memilih item untuk direkomendasikan,
namun recall sedang yang berarti ada banyak item yang relevan tidak muncul di 10 besar rekomndasi. Saran yang bisa dilakukan menambahkan fitur untuk tfidf seperti Description


## Evaluasi
Metrik evaluasi yang digunakan menggunakan RMSE
**RMSE** adalah akar kuadrat dari rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. Metrik ini mengukur seberapa besar rata-rata kesalahan kuadrat prediksi model, memberikan gambaran tentang seberapa jauh prediksi menyimpang dari data sebenarnya. 
**Rumus**

![image](https://github.com/user-attachments/assets/304a2057-aa65-4e48-bb3f-9f1070aa61d5)

1. Visualisasi metrik menggunakan RMSE

![image](https://github.com/user-attachments/assets/e4834a6f-7476-4f8a-a0a6-1495ac5b2767)

## Mendapatkan Rekomendasi Tempat Wisata

![image](https://github.com/user-attachments/assets/ad7f7cc9-9b4d-4f82-91eb-76a2c27db831)

Menampilkan hasil rekomendasi untuk beberapa user dari user ID

![image](https://github.com/user-attachments/assets/d22258b0-2f96-49aa-ad69-fc62c1eb5e79)
![image](https://github.com/user-attachments/assets/f002ca37-8740-4c36-ad56-f402b040a66c)
![image](https://github.com/user-attachments/assets/7f171548-cb15-44a9-b388-33b2c3e01692)

- **Hasil**:
  - RMSE data latih menurun tajam di awal dan stabil di sekitar 0.32-0.33, menunjukkan model belajar dengan baik.
  - RMSE data uji meningkat setelah beberapa epoch dan stabil di sekitar 0.35-0.36, mengindikasikan adanya *overfitting*.
  - Contoh rekomendasi untuk *User_Id* 45:
    - **Destinasi dengan rating tinggi dari pengguna**:
      - Museum Taman Prasasti : Budaya
      - Curug Dago : Cagar Alam
      - Taman Film : Budaya
      - Kampung Pelangi : Taman Hiburan
      - Puncak Kebun Buah Mangunan : Taman Hiburan
    - **Top-10 rekomendasi**:
      - Margasatwa Muara Angke : Cagar Alam
      - Monumen Selamat Datang : Budaya
      - Selasar Sunaryo Art Space : Taman Hiburan
      - Skyrink - Mall Taman Anggrek : Taman Hiburan
      - Kampung Cina : Budaya
      - Masjid Agung Trans Studio Bandung : Tempat Ibadah
      - Museum Tekstil : Budaya
      - Curug Batu Templek : Cagar Alam
      - Teras Cikapundung BBWS : Taman Hiburan
      - Jakarta Planetarium : Taman Hiburan

## Kesimpulan Sistem Rekomendasi
Proyek ini berhasil mengembangkan dua pendekatan sistem rekomendasi untuk destinasi wisata di Indonesia:

Content-Based Filtering: Menggunakan teknik TF-IDF dan cosine similarity untuk menganalisis deskripsi destinasi wisata. Pendekatan ini efektif dalam merekomendasikan tempat-tempat yang memiliki kemiripan konten, seperti kategori dan lokasi, meskipun tanpa data interaksi pengguna sebelumnya.

Collaborative Filtering berbasis Neural Network: Memanfaatkan embedding pengguna dan item dalam model neural network untuk menangkap preferensi pengguna berdasarkan histori interaksi. Pendekatan ini mampu memberikan rekomendasi yang lebih personal dan relevan, terutama ketika tersedia data interaksi yang cukup.

Evaluasi menggunakan metrik RMSE menunjukkan bahwa model collaborative filtering dengan neural network memberikan performa yang lebih baik dalam memprediksi preferensi pengguna dibandingkan pendekatan content-based.

Ke depan, sistem ini dapat dikembangkan lebih lanjut dengan:
1. Menambahkan metadata seperti harga, rating, dan lokasi geografis untuk hybrid recommendation.
2. Mengintegrasikan data real-time dari interaksi pengguna.
3. Menggunakan model deep learning yang lebih kompleks seperti autoencoder atau Transformer-based recommender systems untuk akurasi lebih tinggi.

### Hubungan dengan Business Understanding
- **Problem Statement 1**: Model *content-based filtering* berhasil menyarankan destinasi berdasarkan deskripsi dan kategori, memenuhi kebutuhan untuk rekomendasi berbasis konten.
- **Problem Statement 2**: Model *collaborative filtering* mampu mengidentifikasi preferensi pengguna dari data rating, memberikan rekomendasi personal yang relevan.
- **Problem Statement 3**: Sistem *top-N* rekomendasi efisien, dengan waktu prediksi yang cepat (13 langkah untuk 400+ destinasi) dan hasil yang relevan berdasarkan metrik.
- **Goals**:
  - Model *content-based filtering* berhasil memberikan rekomendasi berdasarkan kemiripan konten, meskipun kurang personal dibandingkan *collaborative filtering*.
  - Model *collaborative filtering* memberikan rekomendasi yang lebih personal, tetapi performanya tergantung pada jumlah data interaksi.
  - Sistem *top-N* efisien dan relevan, cocok untuk aplikasi nyata seperti platform wisata digital.
- **Dampak Solusi**:
  - *Content-Based Filtering*: Efektif untuk pengguna baru (*cold-start problem*) karena tidak memerlukan data interaksi sebelumnya.
  - *Collaborative Filtering*: Memberikan rekomendasi personal yang meningkatkan kepuasan pengguna, tetapi performanya tergantung pada jumlah data interaksi.
  - Kombinasi kedua pendekatan dapat menciptakan sistem hibrida yang lebih robust, meningkatkan engagement dan promosi destinasi wisata.

## Referensi
- M. F. Abdurrafi dan D. H. U. Ningsih, "Content-based filtering using cosine similarity algorithm for alternative selection on training programs," Journal of Soft Computing Exploration, vol. 4, no. 4, pp. 204-212, Desember 2023. [link](https://www.researchgate.net/publication/376963865_Content-based_filtering_using_cosine_similarity_algorithm_for_alternative_selection_on_training_programs?utm_source=chatgpt.com)
- Kaggle Dataset – Indonesia Tourism Destination Dataset. Diakses dari: [Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination/data)
- TensorFlow Documentation – Embedding Layer dan Neural Network for Recommendations: [https://www.tensorflow.org](https://www.tensorflow.org/s/results?q=Embedding%20Layer%20dan%20Neural%20Network%20for%20Recommendations)
