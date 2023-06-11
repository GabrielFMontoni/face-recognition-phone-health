import cv2
import mediapipe as mp

#MediaPipe Drawing Utils sendo atribuido a uma variavel, vai servir pra visualizar os pontos no rosto
mp_drawing = mp.solutions.drawing_utils

#MediaPipe FaceMesh sendo atribuido a uma variavel
mp_face_mesh = mp.solutions.face_mesh

#Variavel pra capturar o video na camera principal (0)
cap = cv2.VideoCapture(0)

#Inicializando FaceMesh
with mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    #Capturando frames enquanto a camera estiver sendo utilizada
    while cap.isOpened():
        #cap.read() retorna sucess e image, é usado pra ler o proximo frame
        #sucess é boleana, se der certo a captura é True, image é o proprio frame
        success, image = cap.read()
        if not success:
            break

        #atribuindo altura e largura da imagem
        image_height, image_width, _ = image.shape

        #Convertendo cores do frame pra rgb (o OpenCV usa BGR como padrao)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Utiliza o frame convertido pro facemesh realizar o processamento, detectando os pontos
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks: #Se conseguiu detectar um rosto vai executar o codigo abaixo
            for face_landmarks in results.multi_face_landmarks: #Para cada ponto:
                for idx, landmark in enumerate(face_landmarks.landmark): #idx é o incide do ponto
                    #landmark é o proprio ponto, contendo xyz, e enumerate é pra fornecer o indice
                    # e o valor (idx e landmark)
                    x = int(landmark.x * image_width) #Capturando X do ponto e multiplicando pra pegar real coordenada
                    y = int(landmark.y * image_height) #Capturando Y do ponto e multiplicando pra pegar real coordenada


                    #Pegar cor RGB dos pontos:
                    rgb = image[y,x] #pegando o valor rgb do ponto de referencia atual, é y,x pela convencao do CV
                    print(f"Cor do ponto {idx} tem as cores {rgb}")

        #Desenhando os pontos de referencia pegos pelo FaceMesh
        mp_drawing.draw_landmarks(
            image,
            #Imagem pra desenhar os pontos
            face_landmarks,
            #Os pontos capturados
            mp_face_mesh.FACEMESH_CONTOURS,
            #Quais areas vão ser desenhadas (FACEMASH_CONTOURS) é o rosto
            #Desenhando os pontos de referencia na cor verde, espessura de 1 e é circular
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            #Desenhando a conexão entre os pontos de cor verde, espessura 1 e marcando que os pontos são redondos
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

        #Abrir janela do OpenCV
        cv2.imshow('Phone Health', image)

        #Se apertar tecla "q" vai sair e parar de rodar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#Liberando a camera e parando o programa
cap.release()
cv2.destroyAllWindows()
