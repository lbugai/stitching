# Инструкция по работе с прогонщиком экспериментов по совмещению трехмерных данных.
Этот репозиторий содержит код методов совмещения, пример структуры папок для работы с данными, тестовые данные для совмещения трехмерных объемов.

## Конфигурация виртуального окружения для работы прогонщика.
Для настройки виртуального окружения в `Windows 10` необходимо проделать следующие шаги:
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

## Запуск прогонщика.



