# Laporan Proyek Machine Learning Terapan 2 - Sistem Rekomendasi Tempat Wisata

Wisnu Al Hussaeni - MC001D5Y1239

## Project Overview
Sektor pariwisata Indonesia menyimpan potensi ekonomi yang besar, namun banyak wisatawan kesulitan menemukan destinasi yang sesuai dengan minat dan kebutuhan mereka karena informasi yang tersebar dan tidak personal. Sistem rekomendasi berbasis machine learning menjadi solusi efektif untuk menyaring informasi dan menyarankan destinasi yang relevan dan menarik bagi setiap individu. Sistem ini tidak hanya meningkatkan kepuasan wisatawan, namun juga membantu destinasi wisata yang kurang dikenal mendapatkan eksposur yang lebih luas.

## Mengapa ini penting?
1. Personalisasi konten menjadi krusial dalam pengambilan keputusan pengguna.
2. Meningkatkan engagement pengguna terhadap platform wisata.
3. Memberikan nilai tambah untuk sektor pariwisata digital Indonesia.

## Business Understanding
### Problem Statements
1. Bagaimana menyarankan destinasi wisata yang relevan berdasarkan konten deskripsi atau nama tempat destinasi?
2. Bagaimana mengidentifikasi preferensi pengguna dari data interaksi sebelumnya untuk memberikan rekomendasi personal?
3. Bagaimana menyusun sistem rekomendasi yang mampu memberikan hasil yang relevan sekaligus efisien dalam waktu pencarian (rekomendasi top-N)?

### Goals
1. Mengembangkan model content-based filtering menggunakan TF-IDF dan cosine similarity untuk menyarankan destinasi berdasarkan deskripsi atau nama tempat yang mirip.
2. Membangun model collaborative filtering berbasis neural network embedding untuk menyarankan destinasi sesuai histori interaksi pengguna.
3. Menghasilkan sistem rekomendasi top-N yang efisien dan akurat untuk digunakan dalam skenario nyata.

### Solution Approach
1. Memanfaatkan deskripsi destinasi wisata.
2. Menggunakan user-item rating matrix.
3. Menggunakan TF-IDF untuk representasi vektor, dan cosine similarity untuk mengukur kemiripan antar destinasi.
4. Membangun neural network dengan embedding layer untuk mempelajari representasi pengguna dan destinasi.

## Data Understanding
Dataset digunakan dari [Kaggle](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination/data)

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

## Data Preprocessing
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

## Model Development Content Based Filtering
1. Membuat data sample

![image](https://github.com/user-attachments/assets/a6b1bacb-fa5e-40bb-8b94-d07298e03be9)

2. Insialisasi TfidfVectorizer

![image](https://github.com/user-attachments/assets/4873bb1c-f13a-44a8-abb6-4e97f3aa51ee)

3. Melakukan fit lalu ditransformasikan ke bentuk matrix

![image](https://github.com/user-attachments/assets/7a3f90d1-450e-4a45-b58f-c2316ec0412a)

4. Membuat dataframe untuk melihat tf-idf matrix

![image](https://github.com/user-attachments/assets/930e24f3-1cdc-4cfb-9019-3ac2b17107fe)

5. Menghitung cosine similarity pada matrix tf-idf

![image](https://github.com/user-attachments/assets/e8cbf1ff-37dc-4fb3-b570-660c9bfe91b6)

6. Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa nama place

![image](https://github.com/user-attachments/assets/204e3780-1a77-4480-98bc-ad7dcbee5add)


## Mendapatkan Rekomendasi
1. Membuat fungsi rekomendasi

![image](https://github.com/user-attachments/assets/fd66073f-81a8-42b0-865a-6435fe4db09f)

2. Mendapatkan rekomendasi place name yang mirip dengan 'Air Mancur Menari'

![image](https://github.com/user-attachments/assets/00474402-35f5-416c-a8ad-16d9ebd29f2c)


## Data Preparation
1. Mengubah PlaceID menjadi list tanpa nilai yang sama

![image](https://github.com/user-attachments/assets/f7d5b2f5-f0aa-467a-a4b2-720f6420ee27)

2. Mengubah userID menjadi list tanpa nilai yang sama

![image](https://github.com/user-attachments/assets/fce406a0-f221-4ef0-8de7-78c355332ddb)

3. Mapping User_Id dan Place_Id ke dataframe user dan place

![image](https://github.com/user-attachments/assets/2f066769-a45f-4b37-a70d-80701a5a3737)

4. Melihat jumlah

![image](https://github.com/user-attachments/assets/6f283e79-2ab9-44ac-946a-5c7765644177)

5. Mengacak dataset untuk keperluan spliting data

![image](https://github.com/user-attachments/assets/4b79bc7d-b373-4b5f-930f-adc74e11df46)

6. Membagi menjadi 80% data train dan 20% data test

![image](https://github.com/user-attachments/assets/8481ca77-d982-4ac4-8274-46329fc2a739)


## Model Development dengan Collaborative Filtering
1. Memuat data tour_rate

![image](https://github.com/user-attachments/assets/82f44583-9441-4427-864c-d08827f81bf6)

2. Proses Training

![image](https://github.com/user-attachments/assets/f29cb7e7-61d7-464a-b679-a24a5464e277)

3. Model menggunakan RecommenderNet

![image](https://github.com/user-attachments/assets/5f0d0096-9a6a-4c64-8147-a5a01fcb4ff9)

4. Menggunakan batch_size=8, epoch=100, verbose=1

![image](https://github.com/user-attachments/assets/af62e189-4e76-4511-8883-dc66c6b45b63)

5. Visualisasi metrik menggunakan RMSE

![image](https://github.com/user-attachments/assets/e4834a6f-7476-4f8a-a0a6-1495ac5b2767)

## Mendapatkan Rekomendasi Tempat Wisata

![image](https://github.com/user-attachments/assets/ad7f7cc9-9b4d-4f82-91eb-76a2c27db831)

Menampilkan hasil rekomendasi untuk beberapa user dari user ID

![image](https://github.com/user-attachments/assets/d22258b0-2f96-49aa-ad69-fc62c1eb5e79)
![image](https://github.com/user-attachments/assets/f002ca37-8740-4c36-ad56-f402b040a66c)
![image](https://github.com/user-attachments/assets/7f171548-cb15-44a9-b388-33b2c3e01692)













