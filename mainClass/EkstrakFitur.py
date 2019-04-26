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
        obj, nonObj = 0,0

        print "Menghitung nilai vektor sample fitur positif (objek yang akan dideteksi)..."
        for img in glob.glob(os.path.join(self.pathPositif, "*")): # load satu-satu data citra objek pada folder datates
            im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            # Hitung nilai HOG untuk mendapatkan fitur objek
            print(orientations, cellSize, cellPerBlock, visualizeHOG, normalizeHOG)
            fitur = hog(im, orientations, cellSize, cellPerBlock) # variabel dari file config
            # fitur = EkstrakWavelet(im).ekstrak()
            vektorObjek.append(fitur)
            obj+=1
        print "Banyak citra objek yang dilatih =",obj
        print "Menghitung nilai vektor sample fitur negatif (citra yang bukan termasuk objek)..."
        for img in glob.glob(os.path.join(self.pathNegatif, "*")): # load satu-satu data citra bukan objek pada folder datates
            im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            # Hitung nilai HOG untuk mendapatkan fitur yang bukan objek
            fitur = hog(im,  orientations, cellSize, cellPerBlock)
            # fitur = EkstrakWavelet(im).ekstrak()
            vektorNonObjek.append(fitur)
            nonObj+=1
        print "Banyak citra bukan objek yang dilatih =",nonObj
        print "Banyaknya vektor yang terbentuk per citra = ",np.size(fitur) # variabel fitur sebagai sample
        # Untuk menampilkan citra hasil perhitungan HOG, visualizeHOG = True
        # cv2.imshow("Contoh citra hasil HOG",fitur[1])
        # cv2.waitKey(0)

        # Mulai training dari data vektor yang telah diperoleh dari ekstraksi fitur
        TrainingData(vektorObjek, vektorNonObjek).startTraining()
