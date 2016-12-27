# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numpy import arctan2, fliplr, flipud

class HOG(object):
	def __init__(self, citra, cellSize=(4, 4), cellPerBlok=(1, 1), signOrientasi=False, nbins=9, visualisasi=False,
				normalisasi=True, flatten=False, sameSize=False):
		super(HOG, self).__init__()
		self.citra = citra
		self.cellSize = cellSize
		self.cellPerBlok = cellPerBlok
		self.signOrientasi = signOrientasi
		self.nbins = nbins
		self.visualisasi = visualisasi
		self.normalisasi = normalisasi
		self.flatten = flatten
		self.sameSize = sameSize
		self.dimensiCitra = citra.shape

	def gradient(self):
		sy, sx = self.dimensiCitra
		if self.sameSize:
			gx = np.zeros(self.dimensiCitra)
			gx[:, 1:-1] = -self.citra[:, :-2] + self.citra[:, 2:]
			gx[:, 0] = -self.citra[:, 0] + self.citra[:, 1]
			gx[:, -1] = -self.citra[:, -2] + self.citra[:, -1]

			gy = np.zeros(self.dimensiCitra)
			gy[1:-1, :] = self.citra[:-2, :] - self.citra[2:, :]
			gy[0, :] = self.citra[0, :] - self.citra[1, :]
			gy[-1, :] = self.citra[-2, :] - self.citra[-1, :]

		else:
			gx = np.zeros((sy-2, sx-2))
			gx[:, :] = -self.citra[1:-1, :-2] + self.citra[1:-1, 2:]

			gy = np.zeros((sy-2, sx-2))
			gy[:, :] = self.citra[:-2, 1:-1] - self.citra[2:, 1:-1]

		magnitude = np.sqrt(gx**2 + gy**2)
		orientation = (arctan2(gy, gx) * 180 / np.pi) % 360
		self.buildHistogram(magnitude, orientation)
		return gx, gy

	def buildHistogram(self, magnitude, orientation):
		sy, sx = magnitude.shape
		csy, csx = self.cellSize

		# checking that the cell size are even
		if csx % 2 != 0:
			csx += 1
			print("WARNING: the cell_size must be even, incrementing cell_size_x of 1")
		if csy % 2 != 0:
			csy += 1
			print("WARNING: the cell_size must be even, incrementing cell_size_y of 1")

		sx -= sx % csx
		sy -= sy % csy
		n_cells_x = sx//csx
		n_cells_y = sy//csy
		magnitude = magnitude[:sy, :sx]
		orientation = orientation[:sy, :sx]
		by, bx = self.cellPerBlok
		print magnitude,"\n"
		print orientation

		# orientation_histogram = interpolate(magnitude, orientation, csx, csy, sx, sy, n_cells_x, n_cells_y, signed_orientation, nbins)

		# if normalise:
		# 	normalised_blocks = normalise_histogram(orientation_histogram, bx, by, n_cells_x, n_cells_y, nbins)
		# else:
		# 	normalised_blocks = orientation_histogram

		# if flatten:
		# 	normalised_blocks = normalised_blocks.flatten()

		# if visualise:
		# #draw_histogram(normalised_blocks, csx, csy, signed_orientation)
		# 	return normalised_blocks, visualise_histogram(normalised_blocks, csx, csy, signed_orientation)
		# else:
		# 	return normalised_blocks
