from imutils import face_utils
from PIL import Image
import numpy as np
import imutils
import dlib
import cv2
import statistics
import os
import ctypes

user32 = ctypes.windll.user32
clear = lambda: os.system('cls')
factor = 0.7 #współczynnik do wykrywania krawędzi tęczówki
strabismus = 0.08 #poniżej brak zeza, powyżej zez występuje


while(1):
	# wybór - wczytanie lub zrobienie zdjęcia
	load_or_make = input("Wpisz 'wczytaj', jeśli chcesz pobrać obraz z istniejącego pliku. Jeśli chcesz zrobić zdjęcie używając kamerki, wpisz 'kamera'\n")
	if load_or_make.lower() == "wczytaj":
		clear()
		input("Obraz musi przedstawiać osobę patrzącą się prosto w obiektyw kamery. Skopiuj obraz do folderu z programem. Nadaj mu nazwę 'Obraz' oraz rozszerzenie .jpg. Gdy to zrobisz, wpisz dowolny tekst i zatwierdź go klawiszem Enter.\n")
		while True:
			image = cv2.imread('Obraz.jpg')
			if image is not None:
				break
			else:
				clear()
				input("Zła nazwa lub rozszerzenie pliku. Skopiuj obraz do folderu z programem. Nadaj mu nazwę 'Obraz' oraz rozszerzenie .jpg. Gdy to zrobisz, wpisz dowolny tekst i zatwierdź go klawiszem Enter.\n")
		break
	elif load_or_make.lower() == "kamera":
		cap = cv2.VideoCapture(0)
		clear()
		child_or_adult = input("Wpisz 'dziecko', jeśli badane jest dziecko. Jeśli badany to dorosły, wpisz 'dorosly' (bez polskich znaków)\n")
		while True:
			#wybór - dziecko czy dorosły
			if child_or_adult.lower() == "dziecko":
				clear()
				print("Niech dziecko patrzy na animację. Gdy będzie nią zajęte, zrób zdjęcie, naciskając spację na włączonym oknie podglądu kamery.\n")
				window_gif = "GIF"
				cv2.namedWindow(window_gif)
				cv2.moveWindow(window_gif, int(user32.GetSystemMetrics(0)/3),0)
				window_camera = "Kamera"
				cv2.namedWindow(window_camera)
				cv2.moveWindow(window_camera, int(user32.GetSystemMetrics(0)/3.3),int(user32.GetSystemMetrics(1)/2.3))
				frame_index = 0
				while True:
					#pętla do wyświetlania gifa oraz obrazu z kamery
					frame_name = f'frame_{frame_index}.jpg'
					if frame_index == 22:
						frame_index=0
					else:
						frame_index = frame_index + 1
					frame = cv2.imread("frames/" + frame_name)
					cv2.imshow(window_gif, frame)
					ret, image = cap.read()
					image = cv2.flip(image, 1)
					cv2.imshow(window_camera, image)
					key = cv2.waitKey(100)
					#wciśnięcie klawisza spowoduje zrobienie zdjęcia
					if (key != -1):
						cap.release()
						cv2.destroyAllWindows()
						break
				break
			elif child_or_adult.lower() == "dorosly":
				window_camera = "Kamera"
				cv2.namedWindow(window_camera)
				cv2.moveWindow(window_camera, int(user32.GetSystemMetrics(0)/3.3),int(user32.GetSystemMetrics(1)/5))
				clear()
				print("Ustaw się na wprost kamery, popatrz się prosto w jej obiektyw i nie przymykaj oczu. Gdy będziesz gotowy, zrób zdjęcie, naciskając spację na włączonym oknie podglądu kamery.\n")
				while(cap.isOpened()):
					ret, image = cap.read()
					image = cv2.flip(image,1)
					cv2.imshow(window_camera, image)
					key = cv2.waitKey(10)
					if (key != -1):
						cap.release()
						cv2.destroyAllWindows()
				break
			else:
				clear()
				child_or_adult = input("Nie wpisałeś swojego wyboru poprawnie. Wpisz 'dziecko', jeśli badane jest dziecko. Jeśli badany to dorosły, wpisz 'dorosly' (bez polskich znaków)\n")
		break
	else:
		clear()
		print("Nie wpisałeś swojego wyboru poprawnie.")


detector = dlib.get_frontal_face_detector() #detektor z biblioteki dlib
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #predyktor wczytany z pliku

image = imutils.resize(image, width=500) #zmiana rozmiaru obrazu
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #zapis obrazu w skali szarości
faces = detector(gray, 1) #zlokalizowanie twarzy na obrazie

for (i, face) in enumerate(faces):

	shape = predictor(gray, face) #zlokalizowanie facial landmarks
	shape = face_utils.shape_to_np(shape)

	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		if name == "right_eye" or name == "left_eye":
			clone = image.copy()
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1) #zaznaczenie oczu

			(x, y, width, height) = cv2.boundingRect(np.array([shape[i:j]]))
			eye = image[y:y + height, x:x + width] #wycięcie obrazów oczu z obrazu

			eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
			eye_gray = imutils.resize(eye_gray, width=250, inter=cv2.INTER_CUBIC)
			eye = imutils.resize(eye, width=250, inter=cv2.INTER_CUBIC) #zapis obrazu oka w skali szarości, zmiana jego rozmiaru

			height, width = eye_gray.shape
			h= height//2
			width_part = width // 25 #niezbędne do werifikacji, czy punkt zakwalifikowany jest krawędzią tęczówki, czy np. rzęsą
			left_list = [eye_gray[h-1][0]] #lista, gdzie zapisujemy punktu z środka wysokości obrazu oka

			for k in range(1,width-1): 
				mean = statistics.mean(left_list) #obliczanie średniej z listy
				if  ((eye_gray[h-1][k] < factor * mean) and (eye_gray[h-1][k] < eye_gray[h-1][k-1] )):
					lp = True
					for i in range(1,width_part): #sprawdzenie zy punkt zakwalifikowany jest krawędzią tęczówki, czy np. rzęsą
						if (eye_gray[h-1][k+i] >= mean):
							lp = False
					if (lp == True):
						left_position = k
						cv2.circle(eye, (k, h), 3, (255, 0, 0), -1) #zaznaczenie krawędzi źrenicy
					break
				left_list.append(eye_gray[h-1][k]) #dodanie punktu, który nie okazał się krawędzią źrenicy do listy

			right_list = [eye_gray[h-1][width-1]]#lista, gdzie zapisujemy punktu z środka wysokości obrazu oka
			for k in range(width-2,0, -1):
				mean = statistics.mean(right_list)#obliczanie średniej z listy
				if  ((eye_gray[h-1][k] < factor * mean) and (eye_gray[h-1][k] < eye_gray[h-1][k+1] )):
					rp = True
					for i in range(1,width_part):#sprawdzenie zy punkt zakwalifikowany jest krawędzią tęczówki, czy np. rzęsą
						if (eye_gray[h-1][k-i] >= mean):
							rp = False
					if (rp == True):
						right_position = k
						cv2.circle(eye, (k, h), 3, (255, 0, 0), -1) #zaznaczenie krawędzi źrenicy
					break
				right_list.append(eye_gray[h-1][k])#dodanie punktu, który nie okazał się krawędzią źrenicy do listy

			middle = (left_position + right_position) // 2 #znalezienie środka źrenicy
			cv2.circle(eye, (middle, h), 3, (0, 0, 255), -1)#zaznaczenie środka źrenicy

			if name == "right_eye":
				clear()
				pupil_right = round((middle/width),3)
				print("Położenie źrenicy w prawym oku osoby badanej: ", pupil_right)				
			elif name == "left_eye":
				pupil_left = round((middle/width),3)
				print("Położenie źrenicy w lewym oku osoby badanej: ", pupil_left)
				difference = round(abs(pupil_left - pupil_right),3)
				print("Różnica między położeniem źrenicy lewej i prawej: ", difference)
				if difference < strabismus:
					print("Nie wykryto zeza u badanego.")
				else:
					print("U badanego wykryto zeza.")
									
			cv2.imshow("Oko", eye)
			cv2.imshow("Obraz", clone)
			cv2.waitKey(0)
			cv2.destroyAllWindows()