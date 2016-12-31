# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import numpy as np
import glob
import os
from config import *
from classifier.svm import SVM
from skimage.feature import hog
from ConfusionMatrix import ConfusionMatrix
import cv2

class TrainingData(object):
    def __init__(self, vektorObjek, vektorNonObjek):
        super(TrainingData, self).__init__()
        self.vektorObjek = vektorObjek
        self.vektorNonObjek = vektorNonObjek
        # tipe klasifier, linear SVM

    def startTraining(self):
        # buat matriks elemen 1 untuk klasifikasi objek dan elemen -1 untuk klasifikasi bukan objek
        kelasObjek = np.ones(len(self.vektorObjek)) # kelas 1 menandakan klasifikasi objek
        kelasNonObjek = np.ones(len(self.vektorNonObjek)) * -1 # kelas -1 menandakan klasifikasi bukan objek

        # push ke stack dari data kelas diatas untuk data train kelas klasifikasi (horizontal stack)
        trainDataY = np.hstack((kelasObjek, kelasNonObjek))
        # print trainDataY,np.size(trainDataY)

        dataVektor = []
        # Push fitur vektor objek ke array dataVektor, buat stack ke array trainDataX (vertical stack)
        for fiturVektor in self.vektorObjek:
            dataVektor.append(fiturVektor)
            trainDataX = np.vstack(dataVektor) # buat stack matriks banyakdata*banyakvektor
            
        # Lanjutkan push fitur vektor yang bukan objek ke array dataVektor, lalu push ke stack trainDataX
        for fiturVektor in self.vektorNonObjek:
            dataVektor.append(fiturVektor)
            trainDataX = np.vstack(dataVektor)
        # print np.size(trainDataX)
        isSave = raw_input("Simpan data ke dataset? (y/n): ")
        if isSave=="y":
            np.savetxt("dataX.csv", trainDataX, delimiter=",")

        # diperoleh array matriks trainDataX yang berukuran banyak data latih * jumlah vektor per data citra
        # dan array matriks trainDataY yang berisi nilai klasifikasi, untuk selanjutnya ditraining menggunakan SVM
        print "Training vektor dengan SVM..."
        clf = SVM(kernel="linear", galat=1e-2, C=0.4)
        clf.fit(trainDataX, trainDataY)

        print "Training sukses."
        print "Spesifikasi model SVM yang telah dilatih: "
        print "-- Nilai bobot:"
        print clf.w
        # plot hasil training data untuk melihat hyperplane-nya
        print "-- Plot Hyperplane:"
        clf.plot_margin(trainDataX[trainDataY==1],trainDataX[trainDataY==-1],clf)

        print "-- Confusion Matrix:"
        ConfusionMatrix(model=clf).printMatrix()

        # Buat folder untuk menyimpan model SVM hasil training
        if not os.path.isdir(os.path.split(folderModel)[0]):
            os.makedirs(os.path.split(folderModel)[0])

        # Simpan model ke folder yang telah dibuat/ada
        joblib.dump(clf, folderModel)
        print "Model classifier saved to {}".format(folderModel)