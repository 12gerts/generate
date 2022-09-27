## Описание
Утилита, которая на основе заданных текстов генерирует новые.

## Состав
* Обучение `train.py`
* Генерация `generate.py`
* Данные `data/`

## Запуск
Файлы следует запускать друг за другом в командной строке.
#### /train.py
```sh
python.exe train.py --input-dir [DIRECTORY_NAME] --model [FILE_NAME]
```
`--input-dir`, `-i` - (необязательный параметр) имя директории с данным. Если этот параметр пропущен, запустите скрипт и следуйте дальнейшим инструкциям, чтобы ввести текст с клавиатуры.  
`--model`, `-m` - название файла, куда будет сохранена модель.  

#### /generate.py
```sh
python.exe generate.py --model [FILE_NAME] --length [LENGTH] --prefix [PREFIX]
```
`--model`, `-m` - название файла, откуда будет загружена модель.  
`--lenght`, `-l` - длина генерируемой последовательности.  
`--prefix`, `-p` - (необязательный параметр) начало предложения, состоящее из одного или нескольких слов. Если параметр пропущен, генерируется случайная последовательность.

## Реализация
Генерация текстов основана на __N-граммной языковой модели__.
1) Чтение текста из файлов в директории или из stdin
2) Токенизация 
3) Составляем N-граммы по заданным текстам в формате:  
`{..., 'слово' : [[вероятность встречаемости], 'следующее слово', 'другое следующее слово, ...], ...}` - для униграмм  
`{..., ('первое слово', 'второе слово') : [[вероятность встречаемости], 'следующее слово', 'другое следующее слово, ...], ...}` - для биграмм  
Вероятность встречаемости вычисляется как количество встреч данной комбинации, нормализованное к 1.
4) Если не был задан префикс, случайным образом выбираем ключ из словаря для начала предложения. По составленной N-граммной модели генерируем последовательность, выбирая следующее по слово по предыдущим с заданной вероятностью.

