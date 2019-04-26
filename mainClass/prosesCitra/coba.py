from hog import HOG
import cv2

img = cv2.imread("tes.pgm",cv2.IMREAD_GRAYSCALE)
cobaa = HOG(img)

x,y = cobaa.gradient()
# print "X =",x
# print "y =",y