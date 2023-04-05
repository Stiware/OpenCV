import cv2
import os

datapath = os.getcwd() + '\Data'
imagePaths = os.listdir(datapath)
print('imagesPaths=',imagePaths)

# Si sale un error en esta parte del codigo ejecutar en cosola: pip install opencv-contrib-python
face_recognizer = cv2.face.LBPHFaceRecognizer_create()


#Se realiza la captura del video con la camara o con un video almacenado en el pc
#cap = cv2.VideoCapture('tuvideo.mp4')
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#Se elige que modelo de reconocimiento se quiere utlizar
face_recognizer.read('modelo_LBPHFace.xml')

#se carga el modelo entrenado al programa
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


while True:
    ret,frame = cap.read()
    if ret == False: break
    #Se pasa la entrada de video a una escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    #se detecta si en el frame existe un rostro o no
    faces = faceClassif.detectMultiScale(gray,1.3,5)
    
    #Por cada rostro encontrado se realiza un cuadrado
    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation = cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        #si el rostro se ha entrenado previamente aparecerá el nombre de la persona en rojo  un recuadro verde
        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(0,0,255),1,cv2.LINE_AA)
        if result[1] < 70:
            cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,+h),(0,255,0),2)
        #Si no se reconoce el rostro aparecerá como desconocido con un recuadro rojo y letras azules
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(255,0,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,+h),(0,0,255),2) 
    #condición que finaliza el programa (tecla esc)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()