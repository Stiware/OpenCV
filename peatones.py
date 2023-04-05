import cv2

#importamos el video de prueba
cap = cv2.VideoCapture('./recursos/in.avi')

#importamos el modelo entrenado para el reconocimiento de personas. ref: https://github.com/codingforentrepreneurs/OpenCV-Python-Series/blob/master/src/cascades/data/haarcascade_fullbody.xml
human_cascade = cv2.CascadeClassifier('./recursos/haarcascade_fullbody2.xml')


while(True):
    # Comienza a leer el video frame por frame
    ret, frame = cap.read()

    # cada farme lo pasamos a una escala de grises y utlizamos el modelo
    # entrenado para saber si lo reconoce como persona o no
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, 1.9, 1)
    
    # a cada humano que reconozca se le dibujará un rectangulo verde al rededor
    for (x,y,w,h) in humans:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
         
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cuando acaba el video se cierra el programa automaticamente
cap.release()
cv2.destroyAllWindows()