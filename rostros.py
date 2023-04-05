import cv2

#inciamos la captura de la cam
cap = cv2.VideoCapture(0)

#importamos el modelo entrenado para el reconocimiento de rostros frontrales. ref: https://github.com/codingforentrepreneurs/OpenCV-Python-Series/blob/master/src/cascades/data/haarcascade_frontalface_default.xml
#importamos el modelo entrenado para el reconocimiento de rostros de perfil . ref: https://github.com/codingforentrepreneurs/OpenCV-Python-Series/blob/master/src/cascades/data/haarcascade_profileface.xml
face_cascadeFront = cv2.CascadeClassifier('./recursos/haarcascade_frontalface_default.xml')
face_cascadeProfile = cv2.CascadeClassifier('./recursos/profile.xml')


while(True):
    # Comienza a la entrada de la camara frame por frame
    ret, frame = cap.read()

    # cada farme lo pasamos a una escala de grises y utlizamos el modelo
    # entrenado para saber si lo reconoce como rostro o no
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascadeFront.detectMultiScale(gray, 1.9, 1)
    face2 = face_cascadeProfile.detectMultiScale(gray, 1.9, 1)
    
    # Cuando reconoce un rostro de manera frontal lo encierra en un cuadrado rojo
    for (x,y,w,h) in face:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    # Cuando reconoce un rostro de perfil lo encierra en un cuadrado azul
    for (x,y,w,h) in face2:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         
    #condición para detener el programa
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()