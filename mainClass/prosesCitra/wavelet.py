import cv2
import numpy as np
import pywt

class EkstrakWavelet(object):
	def __init__(self, citra):
		super(EkstrakWavelet, self).__init__()
		self.citra = citra

	def ekstrak(self):
		imgArray = self.citra
		imgArray =  np.float64(imgArray)
		imgArray/=(255)
		# print imgArray

		matriks4 = pywt.dwt2(imgArray, "haar")
		A, (B, C, D) = matriks4
		x = np.reshape(A, np.product(A.shape))
		# print (x)
		# print np.size(x)
		return x
