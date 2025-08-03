# 📄 Document Scanner and OCR with OpenCV and Tesseract

Этот проект представляет собой простой скрипт для автоматического сканирования документа на изображении (например, фотографии бумажной страницы), корректировки перспективы и распознавания текста с последующим сохранением в PDF.

## 🧰 Используемые технологии

* Python 3
* OpenCV
* imutils
* NumPy
* pytesseract (Tesseract OCR)
* PIL (Pillow)

## 🖼️ Возможности

* Автоматическое определение документа на изображении
* Преобразование перспективы для "сканирования"
* Очистка изображения
* Распознавание текста (русский + английский)
* Сохранение результата в виде PDF-файла с возможностью поиска текста

## 📦 Установка

1. Установите зависимости:

```bash
pip install opencv-python imutils numpy pytesseract pillow
```

2. Установите [Tesseract OCR](https://github.com/tesseract-ocr/tesseract):

* **Windows**: [Скачать инсталлятор](https://github.com/tesseract-ocr/tesseract/wiki#windows)
* **Linux**:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

3. Убедитесь, что указали путь к Tesseract в коде (для Windows):

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\YOUR_USERNAME\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
```

## 🚀 Запуск

```bash
python scan.py --image/-i path_to_your_image.jpg
```

После выполнения в папке с проектом появится файл `scanned_output.pdf`.

## 📝 Пример работы

Исходное изображение:
![000](https://github.com/user-attachments/assets/e2f920a4-6681-47db-af2a-455e19f8da60)


Результат после обработки:
<img width="332" height="487" alt="image" src="https://github.com/user-attachments/assets/56036bf6-ae6a-41c1-a5b4-2302405a9434" />


Сохранённый PDF будет содержать распознанный текст, пригодный для поиска и копирования.

## 📁 Структура проекта

```
project/
├── transform.py
├── scan.py
├── scanned_output.pdf
└── README.md
```

## 🛠️ Заметки

* Убедитесь, что на изображении есть документ с чётко различимыми границами.
* Для наилучшего качества используйте изображения высокого разрешения и хорошего освещения.
* В скрипте используется бинаризация и морфологическая фильтрация для повышения качества OCR.
