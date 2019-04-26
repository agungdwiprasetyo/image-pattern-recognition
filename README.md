# Pengolahan Citra Digital - Pattern Recognition
Penelitian ini menjelaskan tentang program aplikasi pendeteksian suatu objek yang didefinisikan terlebih dahulu dalam sebuah citra. Contoh suatu objek dalam hal ini yaitu ikan, mobil, rumah, tomat, dan lain-lain. Deteksi suatu objek yang digunakan dalam program ini yaitu menggunakan *Histogram of Oriented Gradient* (HOG) untuk mengekstraksi ciri dari objek dan bukan objek/*background*, serta proses pelatihan (*learning*) menggunakan *Support Vector Machine* (SVM) untuk melatih data yang telah diekstrak yang selanjutnya dilakukan proses klasifikasi antara objek dan bukan objek. Objek yang akan dideteksi dalam penelitian ini adalah **ikan**.

## *Requirements*
Program ini menggunakan bahasa pemrograman Python versi 2.7 dan ditulis dalam struktur *Object Oriented Programming* (OOP). Tujuan dikembangkan dalam struktur OOP yaitu supaya mempermudah pembacaan program dengan membagi kedalam modul-modul/kelas. Program ini dikembangkan dalam sistem operasi Linux.

Jika baru menginstal python, instal ```python-pip``` dan ```python-dev``` dahulu untuk mempermudah menginstal library-library yang dibutuhkan selanjutnya.
```sh
$ sudo apt-get install python-pip python-dev
```
Instal OpenCV pada python, tutorial buka link ini ->
[Install OpenCV](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/)

Jika menggunakan MacOS:
```sh
$ brew install opencv3
```

Instal library yang dibutuhkan yaitu: ```numpy```, ```scikit-image```, ```matplotlib``` (untuk plot *hyperplane* pada SVM), ```cvxopt``` (untuk operasi matriks dan masalah optimasi pada SVM):
```
$ sudo pip install numpy
$ sudo pip install scikit-image
$ sudo pip install matplotlib
$ sudo pip install cvxopt
```
Jalankan program:
```sh
$ python start.py
``` 
atau bisa juga (jika menggunakan Terminal Linux):
```sh
$ ./start.py
```
Jika masih terdapat error, instal library lain yang muncul di pesan error tersebut pada python:
```sh
$ sudo pip install [nama library]
```


## Tahapan Program
Secara umum tahapan dalam program ini yaitu pengumpulan data latih citra, praproses citra, ekstraksi ciri masing-masing data citra, proses *training* hasil ekstraksi ciri untuk pengenalan pola, dan *testing* data.
![tahap](https://github.com/agungdwiprasetyo/project-ppcd/raw/master/imagesMarkdown/tahapanPr.png)

1. **Pengumpulan Data dan Praproses Citra**: Citra yang sudah dikumpulkan disamakan ukuran resolusinya menjadi 200x80. Data latih positif (objek yang akan dideteksi) sebanyak 273 citra dan data latih negatif (data yang bukan termasuk objek) sebanyak 115 citra, yang terdapat dalam folder ```data/datalatih/ikan/TrainImages```). Lalu, dilakukan perubahan warna piksel pada citra menjadi *grayscale*. Hal itu dikarenakan untuk membuat citra menjadi satu *channel* agar memudahkan proses pelatihan citra.
**Jumlah keseluruhan data latih yaitu sebanyak 388 data citra**.
![data](https://github.com/agungdwiprasetyo/project-ppcd/raw/master/imagesMarkdown/datalatih.png)

2. **Ekstraksi Ciri**: Metode ekstraksi ciri yang digunakan dalam program ini adalah *Histogram of Oriented Gradient* (HOG), yang bekerja dalam daerah spasial. Ada beberapa metode lain untuk mengekstraksi ciri dari citra yaitu **Wavelet** yang bekerja dalam daerah frekuensi. HOG adalah sebuah metode untuk pendeteksian objek dengan menghitung nilai gradien dalam daerah tertentu pada citra. Distribusi gradien menunjukkan karakteristik dari setiap citra. Karateristik ini diperoleh dengan membagi citra ke dalam daerah kecil yang disebut *cell*. Setiap *cell* disusun sebuah histogram dari sebuah gradien. *Cell* biasanya memiliki ukuran 4x4 piksel sedangkan *block* memiliki ukuran 2x2 *cell* atau 8x8 piksel. Vektor gradien dari suatu piksel dihitung dengan mengurangi nilai piksel tetangga kiri dikurang nilai piksel tetangga kanan serta nilai piksel tetangga atas dikurang nilai piksel tetangga bawah. Tahap normalisasi dilakukan pada vektor gradien yang *outlier*. Normalisasi dilakukan dengan melakukan perkalian pada nilai tertentu terhadap tiap piksel. Hasil *feature* dari HOG diubah menjadi *feature vector* yang selanjutnya akan dilatih dengan SVM untuk pengenalan pola. 
![ciri](https://github.com/agungdwiprasetyo/project-ppcd/raw/master/imagesMarkdown/ekstraksiCiri.png)
**Banyaknya vektor per citra hasil ekstraksi yaitu berjumlah 14904, sehingga ukuran *datasets* yang akan dilatih oleh SVM yaitu sebesar 388x14904**

3. **Pengenalan Pola dengan *Support Vector Machine* (SVM)**: Program ini menggunakan dataset citra yang diambil melalui *google image* sebagai citra *template* yang nantinya akan dibandingkan dengan citra masukan. Metode algoritma yang digunakan adalah *Support Vector Machine* (SVM). Algoritma tersebut merupakan teknik untuk melakukan prediksi baik dalam kasus klasifikasi atau regresi. Konsep dasar SVM merupakan kombinasi dari teori-teori komputasi , seperti *margin hyperplane*, kernel, dan konsep-konsep pendukung lainnya. SVM pun masih berada dalam satu kelas dengan *Artificial Neural Network* (ANN) dalam hal fungsi dan kondisi permasalahan yang bisa diselesaikan. Dalam program ini, teknik SVM digunakan sebagai *classifier* untuk membandingkan citra masukan dengan citra *template*. Kernel yang digunakan pada SVM adalah linear, (kernel lainnya yaitu *Polynomial* dan *Gaussian*). Data untuk objek yang positif dan negatif dapat dipisahkan secara tegas. Pendekatan yang digunakan adalah *Quadratic Programming* (QP). QP memiliki fungsi objektif yang kuadratik dengan kendala yang linear. Untuk optimisasi fungsi objektif digunakan *Lagrange Multipliers*. Dalam menyelesaikan permasalahan-permasalahan optimasi *quadratic programming* pada matriks yang terbentuk, digunakan library ```cvxopt``` pada python (*dapat dilihat di file program pada folder* ```mainClass/classifier/svm.py```). Metode SVM dalam program aplikasi ini dibuat dalam kelas sendiri, dan **tidak menggunakan library**. Ada opsional lain untuk menggunakan metode SVM ini, yaitu dengan menginstal library ```scikit-learn```. *Library* ```scikit-learn``` dalam python tidak hanya SVM saja, tetapi ada algoritma *machine learning* lainnya seperti ANN, Naive Bayes, dan K-Nearest Neighbors (KNN).

	**Untuk proses klasifikasi, data vektor yang termasuk objek diberi label kelas 1 sedangkan data vektor yang bukan termasuk objek diberi label kelas -1.**
    
	**Dari data training yang telah diproses dengan SVM, diperoleh gambar *hyperplane* dan *margin* seperti berikut:**
![data](https://github.com/agungdwiprasetyo/project-ppcd/raw/master/imagesMarkdown/hyperplane.png)

	**Berikut hasil evaluasi untuk akurasi dan *confusion matriks* dari model SVM yang telah dilatih:**
![akurasi](https://github.com/agungdwiprasetyo/project-ppcd/raw/master/imagesMarkdown/confussionMatrix.png)

## Hasil
Proses pendeteksian objek dalam citra, program melakukan *sliding template* (kotak persegi panjang putih yang akan berjalan dari kiri ke kanan lalu dari atas ke bawah) yang berukuran sama dengan data latih (200x80 px). Ketika proses *sliding* ini berjalan maka dilakukan proses perhitungan vektor gradien dari *template* saat *sliding* tersebut menggunakan HOG. Hasil perhitungan vektor tersebut kemudian dites ke model SVM yang telah disimpan untuk mendapatkan nilai prediksi. Bila nilai tes prediksi sama dengan 1 maka *template* tesebut masuk ke kelas positif sehingga *template* saat *sliding* ditandai sebagai objek (**ditandai dengan persegi panjang hitam pada gambar dibawah**) dan atribut dari *template* tersebut disimpan. Setelah proses *sliding* selesai, maka dilakukan *scalling-down* pada citra lalu dilakukan kembali proses *sliding*. Hal ini dilakukan untuk mendeteksi objek yang kemungkinan lebih besar dari ukuran citra pada datalatih. Untuk proses *sliding template* dapat digambarkan sebagai berikut:

![sliding](https://github.com/agungdwiprasetyo/project-ppcd/raw/master/imagesMarkdown/sliding1.png)

Langkah selanjutnya yaitu dengan menormalisasi kotak-kotak hitam yang saling menumpuk sehingga diperoleh satu kotak saja (menggunakan *Non-Maxima Suppression*). Hasil akhir pendeteksian objek dalam citra dapat dilihat pada gambar dibawah ini.

![hasil](https://github.com/agungdwiprasetyo/project-ppcd/raw/master/imagesMarkdown/hasil1.png)

Dapat terlihat ada beberapa ikan yang tidak terdeteksi. Hal ini karena kurang beragamnya data latih ikan yang di-*training* dengan SVM.

## Evaluasi
Pengembangan program aplikasi dengan menggunakan SVM membutuhkan banyak data latih dalam prosesnya. Data latih yang besar dan beragam dapat meningkatkan akurasi dalam pendeteksian objek. Ekstraksi ciri menggunakan HOG membutuhkan citra objek dari berbagai sisi sehingga ciri yang diekstrak lebih beragam. Ciri ini mampu menghasilkan model yang lebih akurat ketika dilakukan pelatihan dengan SVM. Data latih objek yang digunakan pada program ini hanya citra ikan yang tampak dari samping, akurasi yang dihasilkan sebesar 90.67%. Untuk meningkatkan akurasi diperlukan data latih yang lebih beragam.

## Saran
Saran dalam penelitian ini adalah penggunaan data latih yang lebih banyak serta beragam untuk meningkatkan akurasi dari model SVM. Penambahan data latih dilakukan pada data objek dan bukan objek. Dilakukan *image enhancement* pada data uji sehingga antar objek dan bukan objek dapat dibedakan dengan jelas.

### 
```python
#!/usr/bin/env python
s = "'Let’s go to invent tomorrow instead of worrying about what happened yesterday' – Steve Jobs"
print s
```