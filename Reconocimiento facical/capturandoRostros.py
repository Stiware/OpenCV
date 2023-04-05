import cv2
import os
import imutils

#definimos el nombre de la persona a reconocer y la ruta donde se almacenarán las imagenes obtenidas
personName = input('Ingrese el nombre de la persona a la que se le escaneará el rostro: \n')
dataPath = os.getcwd() + "\Data"
personPath = dataPath+ "/" +personName

#si la ruta no existe se crea una
if not os.path.exists(personPath):
    print('carpeta creada: ',personPath)
    os.makedirs(personPath)
    
#cap = cv2.VideoCapture('tuVideo.mp4')
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#se carga el modelo de reconocimiento facial que se utlizará y se define el numero de imagenes a guardar
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 400

while True:
    ret, frame = cap.read()
    if ret == False: break
    #se redimencina el tamaño de cada frame capturado por la camara 
    frame = imutils.resize(frame, width=640)
    #los frames se pasan a una escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    #Se detectan los rostros con el clasificador seleccionado anteriormente
    faces = faceClassif.detectMultiScale(gray,1.3,5)
    
    #Utilizamos un blucle para mostrar cuando se detecta un rostro y cuando no
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,+h),(0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation = cv2.INTER_CUBIC)
        #al detectar el rostro se guarda la imagen en su respectiva carpeta hasta llegar a la cantidad de imagenes establecido anteriormente
        cv2.imwrite(personPath+'/rostro_{}.jpg'.format(count),rostro)
        count = count + 1
    cv2.imshow('frame',frame)
    
    k = cv2.waitKey(1)
    if k == 27 or count >= 600:
        print('Captura de rostros finalizada')
        break
    
cap.release()
cv2.destroyAllWindows()