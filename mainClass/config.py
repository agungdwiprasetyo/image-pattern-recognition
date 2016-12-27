'''
Set variabel utama yang digunakan pada kelas
'''

# Variabel yang digunakan dalam proses HOG
resolusiTemplate = [200, 80]
stepSize = [20, 20]
orientations = 9
cellSize = [8, 8]
cellPerBlock = [3, 3]
"""
visualizeHOG = True: ada dua array yg terbentuk di fitur, array ke-0 sebagai data vektor, array ke-1 untuk 
menampilkan citra vektor tersebut (pake cv2.imshow)
visualizeHOG = False: hanya satu array yang terbentuk, yaitu array vektor saja
"""
visualizeHOG = False
normalizeHOG = True

# Folder data latih
# Ikan
tesObjek = "data/datalatih/Ikan/TrainImages/pos"
tesNotObjek = "data/datalatih/Ikan/TrainImages/neg"

tesPositif = "data/datalatih/Ikan/TestAkurasi/pos"
tesNegatif = "data/datalatih/Ikan/TestAkurasi/neg"

# Folder tempat vektor hasil ekstrak ciri dan model
fiturObjek = "data/fitur/pos"
fiturBukanObjek = "data/fitur/neg"
folderModel = "data/models/svm.model"

# nilai thresold untuk proses NMS
threshold = 0.2
