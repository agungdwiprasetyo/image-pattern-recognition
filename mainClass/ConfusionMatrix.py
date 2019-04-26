# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import numpy as np
import glob
import os
from config import *
from classifier.svm import SVM
from skimage.feature import hog
import cv2

class ConfusionMatrix(object):
	def __init__(self, model):
		super(ConfusionMatrix, self).__init__()
		self.model = model
		self.matPositif = []
		self.matNegatif = []
		self.truePositif, self.falsePositif, self.trueNegatif, self.falseNegatif = 0,0,0,0

	def printMatrix(self):
		# buat matriks hasil prediksi untuk kelas positif
		jumlahData = 0
		for imageTestPositif in glob.glob(os.path.join(tesPositif, "*")): # variabel tesPositif dari file config.py
			img = cv2.imread(imageTestPositif, cv2.IMREAD_GRAYSCALE)
			fitur = hog(img, orientations, cellSize, cellPerBlock)
			prediksi = self.model.predict(fitur)
			self.matPositif.append(prediksi)
			jumlahData+=1

		for imageTestNegatif in glob.glob(os.path.join(tesNegatif, "*")): # variabel tesNegatif dari file config.py
			img = cv2.imread(imageTestNegatif, cv2.IMREAD_GRAYSCALE)
			fitur = hog(img, orientations, cellSize, cellPerBlock)
			prediksi = self.model.predict(fitur)
			self.matNegatif.append(prediksi)
			jumlahData+=1

		for i in range(len(self.matPositif)):
			if self.matPositif[i]==1:
				self.truePositif+=1
			else:
				self.falsePositif+=1

		for i in range(len(self.matNegatif)):
			if self.matNegatif[i]==-1:
				self.trueNegatif+=1
			else:
				self.falseNegatif+=1

		print "               | Prediksi Positif | ","Prediksi Negatif |"
		print "Aktual Positif |      ",self.truePositif,"                 ",self.falsePositif,"          "
		print "               |------------------|-------------------|"
		print "Aktual Negatif |      ",self.falseNegatif,"                 ",self.trueNegatif,"          \n"

		akurasi = (np.float(self.truePositif+self.trueNegatif)/jumlahData)*100
		print "\tAkurasi =",akurasi,"%\n"