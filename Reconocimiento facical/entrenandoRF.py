import cv2
import os
import numpy as np

datapath = os.getcwd() +"\Data"
peopleList = os.listdir(datapath)

labels = []
faceData = []
label = 0

#bucle que almacena todas las rutas de las imagenes por persona y las almacena en un arreglo
for nameDir in peopleList:
    personPath = datapath + '/' + nameDir
    print("Leyendo las imagenes")
    
    for fileName in os.listdir(personPath):
        print("Rostros: ", nameDir + '/' +fileName)
        labels.append(label)
        faceData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
    label = label + 1
#

#Se elige el modelo con el cual se quiere entrenar
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(faceData,np.array(labels))

#Almacenando el modelo obtenido sobreescribiendo el archivo ya existente si es el caso
face_recognizer.write('modelo_LBPHFace.xml')
print("Modelo Almacenado...")

#cv2.destroyAllWindows

