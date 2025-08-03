# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\mark6\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# загрузить изображение и вычислить отношение старой высоты
# на новую высоту, клонируйте его и измените размер
image = cv2.imread(args["image"])

if image is None:
    print(f"Не удалось загрузить изображение: {args['image']}")
    exit(1)

ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# преобразовать изображение в оттенки серого, размыть его и найти края
# на изображении
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# найти контуры на изображении с кромками, оставив только
# самые большие и инициализируем контур экрана
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# цикл по контурам
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# если наш приближенный контур имеет четыре точки, то мы
	# можно предположить, что мы нашли наш экран
	if len(approx) == 4:
		screenCnt = approx
		break

# Проверка, найден ли контур
if screenCnt is None:
    print("Не удалось найти контур документа.")
    exit(1)

# применяем преобразование по четырем точкам для получения нисходящего изображения
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

kernel = np.ones((1, 1), np.uint8)
warped_clean = cv2.morphologyEx(warped, cv2.MORPH_OPEN, kernel)

pil_img = Image.fromarray(warped_clean)

# Сохранение PDF с распознанным текстом
pdf = pytesseract.image_to_pdf_or_hocr(pil_img, extension="pdf", lang="rus+eng")
with open("scanned_output.pdf", "wb") as f:
    f.write(pdf)