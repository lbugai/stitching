# Инструкция по работе с прогонщиком экспериментов по совмещению трехмерных данных.
Этот репозиторий содержит код метода совмещения [SITK](https://simpleitk.org/), пример структуры папок для работы с данными, тестовые данные, инструкцию по работе и запуску тестового примера на `Windows 10`.

## Конфигурация виртуального окружения для работы прогонщика.
Для настройки виртуального окружения необходимо проделать следующие шаги:
1. Создать чистое виртуальное окружение с помощью [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html):
```shell
python –m venv venv_name
```
 Также можно воспользоваться [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) .
 
 2. Активировать окружение:
```shell
path/to/venv_name/Scripts/activate.bat
```

3. Запустить в терминале из папки с файлом `required_libraries.txt` команду
```shell
pip install -r required_libraries.txt
``` 

## Работа с прогонщиком.
### Структура папок
Для работы с прогонщиком потребуется создать структуру папок и файлов следующего вида (пример такой структуры представлен в репозитории, поэтому его можно клонировать и использовать после настройки конфигов):
- runner
  - experiment_runner.py
  - metrics.py
  - parser.py
  - registration_gui.py
  - required_libraries.txt
  - results_visualization.py
  - sample_creator.py
  - SITK.py
- experiments
  - algorithms_configs
    - sitk_config.json
  - exp0
     - config_exp0.json
  - exp1
     - config_exp1.json
  - test_data
     - data1
       - ...
     - data2
       - ...
  


