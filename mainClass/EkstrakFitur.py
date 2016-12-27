# -*- coding: utf-8 -*-
from skimage.feature import hog
import cv2
import glob
import os
from config import * # import variabel
import numpy as np
from TrainCitra import TrainingData
from prosesCitra.wavelet import EkstrakWavelet

class EkstrakFitur(object):
    def __init__(self):
        super(EkstrakFitur, self).__init__()
        self.pathPositif = tesObjek
        self.pathNegatif = tesNotObjek
        # tipe deskriptor = HOG

    def startExtract(self):
        # Buat array untuk menampung data vektor hasil ekstraksi fitur data latih objek dan bukan objek
        vektorObjek = []
        vektorNonObjek = []

        print "Menghitung nilai vektor sample fitur positif (objek yang akan dideteksi)..."
        for img in glob.glob(os.path.join(self.pathPositif, "*")): # load satu-satu data citra objek pada folder datates
            im = cv2.imread(img, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            # Hitung nilai HOG untuk mendapatkan fitur objek
            fitur = hog(im, orientations, cellSize, cellPerBlock, visualizeHOG, normalizeHOG) # variabel dari file config
            # fitur = EkstrakWavelet(im).ekstrak()
            vektorObjek.append(fitur)
        print "Menghitung nilai vektor sample fitur negatif (citra yang bukan termasuk objek)..."
        for img in glob.glob(os.path.join(self.pathNegatif, "*")): # load satu-satu data citra bukan objek pada folder datates
            im = cv2.imread(img, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            # Hitung nilai HOG untuk mendapatkan fitur yang bukan objek
            fitur = hog(im,  orientations, cellSize, cellPerBlock, visualizeHOG, normalizeHOG)
            # fitur = EkstrakWavelet(im).ekstrak()
            vektorNonObjek.append(fitur)

        print "Banyaknya vektor yang terbentuk per citra = ",np.size(fitur)
        # Untuk menampilkan citra hasil perhitungan HOG, visualizeHOG = True
        # cv2.imshow("Contoh citra hasil HOG",fitur[1])
        # cv2.waitKey(0)

        # Mulai training dari data vektor yang telah diperoleh dari ekstraksi fitur
        TrainingData(vektorObjek, vektorNonObjek).startTraining()
