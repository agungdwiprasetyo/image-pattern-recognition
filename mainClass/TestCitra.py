# -*- coding: utf-8 -*-
from skimage.transform import pyramid_gaussian
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
from config import *
import numpy as np
from prosesCitra.nms import nms
from prosesCitra.wavelet import EkstrakWavelet

class DeteksiObjek(object):
    def __init__(self, objekCitra, downscale=2.5, visualisasi=True):
        super(DeteksiObjek, self).__init__()
        self.objekCitra = objekCitra
        self.downscale = downscale
        self.visualisasi = visualisasi
        self.minWindowSize = resolusiTemplate
        self.stepSize = stepSize
        self.kecepatanSliding = 1
        self.toleransiTerdeteksi = 0.02 # set batas nilai confidence dari objek yang terdeteksi berdasarkan prediksi dari SVM

    def sliding_window(self, image, window_size, stepSize):
        for y in xrange(0, image.shape[0], stepSize[1]):
            for x in xrange(0, image.shape[1], stepSize[0]):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

    def startDeteksiObjek(self):
        # Baca citra yang akan dideteksi objeknya
        im = cv2.imread(self.objekCitra, cv2.IMREAD_GRAYSCALE)
        # Load model SVM
        clf = joblib.load(folderModel)

        detections = []
        scale = 0
        # Buat scaling/memperkecil citra yang dites, tujuannya supaya objek yg besar bisa dideteksi (pake fungsi pyramid gaussian dgn parameter ukuran scalling)
        for im_scaled in pyramid_gaussian(im, downscale=self.downscale):
            propertiTerdeteksi = [] # berisi lokasi objek (x,y), nilai confidence, dan ukuran image saat objek terdeteksi

            if im_scaled.shape[0] < self.minWindowSize[1] or im_scaled.shape[1] < self.minWindowSize[0]:
                break # kalo citra-tes scale-nya udah lebih kecil/sama dengan ukuran kotak sliding windows, proses scan berhenti

            # start sliding windows, untuk scan bagian image sesuai ukuran rectangle/resolusi dari data latih
            for (x, y, im_window) in self.sliding_window(im_scaled, self.minWindowSize, self.stepSize):
                if im_window.shape[0] != self.minWindowSize[1] or im_window.shape[1] != self.minWindowSize[0]:
                    continue
                # Hitung hog dari bagian sliding windows, variabel parameter dari file config.py
                fitur = hog(im_window, orientations, cellSize, cellPerBlock)
                # fitur = EkstrakWavelet(im_window).ekstrak()
                fitur = np.array(fitur).reshape((1, -1))
                prediksi = clf.predict(fitur)
                if prediksi == 1:
                    # Tandai objek yang terdeteksi, dengan menyimpan lokasi objek
                    print  "Objek terdeteksi:: Lokasi -> ({}, {})".format(x, y)
                    # print "Scale ->  {} | Akurasi {} \n".format(scale,clf.akurasi(fitur))
                    detections.append((x, y, clf.akurasi(fitur), int(self.minWindowSize[0]*(self.downscale**scale)), int(self.minWindowSize[1]*(self.downscale**scale))))
                    propertiTerdeteksi.append(detections[-1]) # append detections[-1] buat push array dari indeks belakang array detections

                if self.visualisasi:
                    clone = im_scaled.copy()
                    for x1, y1, _, _, _  in propertiTerdeteksi: # subarray yg diambil hanya posisi objek yg terdeteksi (x1,Y1), yg lain diabaikan
                        cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 + im_window.shape[0]), (0, 0, 0), thickness=2)
                    cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y + im_window.shape[0]), (255, 255, 255), thickness=2)
                    cv2.imshow("Proses scan untuk mendeteksi objek", clone)
                    cv2.waitKey(self.kecepatanSliding)
                    # print "masuk massss"

                # print detections
            # Perkecil image, agar objek yg besar bisa terdeteksi
            scale+=1

        clone = im.copy()
        # for (x, y, _, panjang, lebar) in detections:
        #     cv2.rectangle(im, (x, y), (x+panjang, y+lebar), (0, 0, 0), thickness=2)
        # cv2.imshow("Deteksi mentah, rectangle masih menumpuk", im)
        # cv2.waitKey()

        # Perform Non Maxima Suppression, buat ngilangin rectangle yg menumpuk
        detections = nms(detections, threshold)

        # Tandai dengan rectangle untuk deteksi akhir objek
        jumlahObjek = 0
        for (x, y, toleransi, panjang, lebar) in detections:
            if toleransi>self.toleransiTerdeteksi:
                print "Lokasi objek yang terdeteksi -> ({},{})".format(x,y)
                cv2.rectangle(clone, (x, y), (x+panjang,y+lebar), (255, 0, 0), thickness=3)
                jumlahObjek+=1

        print "Jumlah objek yang terdeteksi ada "+str(jumlahObjek)+" buah."
        cv2.imshow("Deteksi Akhir, menghilangkan rectangle yang saling menumpuk", clone)
        cv2.waitKey()
