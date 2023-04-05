import cv2
import numpy as np

#Inicia la camara capturando en modo video
cap = cv2.VideoCapture(0)

#Determinar rangos del color rojo
redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([8, 255, 255], np.uint8)

redBajo2=np.array([175, 100, 20], np.uint8)
redAlto2=np.array([179, 255, 255], np.uint8)

#Determinar rangos del color verde
greenBajo = np.array([35, 100, 20], np.uint8)
greenAlto = np.array([85, 255, 255], np.uint8)

#Determinar rangos del color azul
blueBajo=np.array([100, 100, 20], np.uint8)
blueAlto=np.array([130, 255, 255], np.uint8)

while True:
  ret,frame = cap.read()
  if ret==True:
    #se transforma el video capturado de RGB a HSV
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #se define un primer rango para el color rojo
    maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
    
    #se define un segundo rango para el color rojo
    maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)
    
    #se unen ambos rangos para generar una mascara que identifique el color rojo
    maskRed = cv2.add(maskRed1, maskRed2)
    
    #se crean las respectivas mascaras del color verde y azul
    maskGreen = cv2.inRange(frameHSV, greenBajo, greenAlto)
    maskBlue = cv2.inRange(frameHSV, blueBajo, blueAlto)
    
    #Se transforma de una escala de grises a una escala de colores rbg donde se pueda visualizar cada color por separado
    maskRedvis = cv2.bitwise_and(frame, frame, mask= maskRed)
    maskBluevis = cv2.bitwise_and(frame, frame, mask= maskBlue)
    maskGreenvis = cv2.bitwise_and(frame, frame, mask= maskGreen)
    
    #ventana original
    #cv2.imshow('frame', frame)
    #ventana que muestra solo el rojo
    cv2.imshow('frameRED', maskRedvis)
    #ventana que muestra solo el azul
    cv2.imshow('frameBlue', maskBluevis)
    #ventana que muestra solo el verde
    cv2.imshow('frameGreen', maskGreenvis)
    #condición para detener el programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()

