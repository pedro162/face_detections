import cv2

detector_face = cv2.CascadeClassifier('/home/pedro/PycharmProjects/DeteccaoDeFaces/arquivos_curos_visao_computacional/drive-download-20230813T171824Z-001/Cascades/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)#Primeiro dispositivo na máquina, camera

while True:
    #Captura frame a frame
    ok, frame = video_capture.read()
    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    deteceos = detector_face.detectMultiScale(imagem_cinza, minSize=(200, 200))

    #Dezenha o retângulo
    for (x, y, w, h) in deteceos:
        print(w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #Mostra o resultado no vídeo
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

"""
    OpenCV documentation
    https://docs.opencv.org/4.x/
    
    Dlib documentation
    http://dlib.net/python/index.html
    
    Face Detection: A Literature Review
    http://www.ijirset.com/upload/2017/july/92_Face.pdf
    
    Cascade classifier
    https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
    
    HOG classifier
    https://hal.inria.fr/inria-00548512/document
    
    Building your own object detector with OpenCV (Minito interessante)
    https://medium.com/@vipulgote4/guide-to-make-custom-haar-cascade-xml-file-for-object-detection-with-opencv-6932e22c3f0e

    Building your own object detector with Dlib
    https://learnopencv.com/training-a-custom-object-detector-with-dlib-making-gesture-controlled-applications/
"""