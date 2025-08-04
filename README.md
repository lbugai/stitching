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
- `runner`
  - `experiment_runner.py`
  - `metrics.py`
  - `parser.py`
  - `registration_gui.py`
  - `required_libraries.txt`
  - `results_visualization.py`
  - `sample_creator.py`
  - `SITK.py`
- `experiments`
  - `algorithms_configs`
    - `sitk_config.json`
  - `exp0`
     - `config_exp0.json`
  - `exp1`
     - `config_exp1.json`
  - `test_data`
     - `data1`
       - `...`
     - `data2`
       - `...`
  
В папке `runner` (может иметь любое название) содержатся файлы для запуска прогонщика, а также файл [`required_libraries.txt`](https://github.com/lbugai/stitching/blob/main/required_libraries.txt) с необходимыми библиотеками.

Папка `experiments` (может иметь любое название) - это основная папка, содержащая все эксперименты `exp`. `Все эксперименты необходимо запускать из папки experiments`: для каждого эксперимента с номером `N` создается папка `expN`, туда помещается конфиг эксперимента `config_expN.json`. Количество экспериментов неограничено.

Также в `experiments` есть папка `algorithms_configs` (может иметь любое название), содержащая конфиги методов, их может быть сколько угодно. Вообще говоря, конфиги могут иметь любое расположение. `Расположение конфигов методов указывается в конфиге эксперимента`, но для удобства папка `algorithms_configs` с ними добавлена в `experiments`. 

### Запуск тестового примера.
В репозитории содержатся тестовые данные, конфиги, результат работы прогонщика.
Ниже будет представлено объяснение процедуры настройки и запуска прогонщика совмещения на примере структуры папок и данных в репозитории.

Для начала необходимо выкачать или клонировать данный репозиторий.
`Для запуска тестового примера необходимо распаковать` [`тестовые данные`](https://github.com/lbugai/stitching/tree/main/experiments/test_data), содержащие реконструкцию грецкого ореха по слоям в изображениях формата `.tif`, и удалить архивы.
`Обратите внимание, изображения должны называться по паттерну prefix_0000.tif, prefix_0001.tif, ..., а не prefix_0.tif, prefix_1.tif, ...`
Для переименования файлов в репозитории содержится скрипт `utils.py` (ИЛИ ДОБАВЛЮ ОПЦИЮ В КОНФИГ КОТОРАЯ ЭТО УЧИТЫВАЕТ, ЛУЧШЕ НАВЕРНОЕ НЕ МЕНЯТЬ ДАННЫЕ ПОЛЬЗОВАТЕЛЯ, ХЗ)

В репозитории представлено два эксперимента с парами данных в первом `nut` и `nut_rotated` (повернутый `nut`) и во втором `nut` и `nut_rotated` (увеличенный `nut`).

Для запуска примеров необходимо настроить конфиги экспериментов.


<details><summary>Описание параметров конфига эксперимента 0.</summary>

```json
{
    "path_to_main_folder" : "/path/to/experiments/", # путь до основной папки experiments
    "exp" : 0, # для каждого эксперимента создается папка exp с номером, туда копируется этот конфиг и редактируется. В конфиге в папке expN этот параметр тоже должен быть N, ИНАЧЕ ЭКСПЕРИМЕНТ С УКАЗАННЫМ ЗДЕСЬ НОМЕРОМ ПЕРЕЗАПИШЕТСЯ.
    "algorithm_name" : "sitk",
    "algorithm_help" : {
        "1" : "sitk"
    },
    "algorithm_execution_parameters_path": "/path/to/experiments/algorithms_configs/sitk_config.json", # путь до конфига с настройкой параметров метода, для каждого эксперимента можно сделать такой конфиг и хранить например в папке exp
    "algorithm_executable_path": "/path/to/runner/SITK.py", # путь до скрипта с методом 
    "alg_interpreter_path": "/path/to/venv_name/Scripts/python.exe", # путь до файла python.exe вашего окружения

    "VolumeLoadingMode": "TwoVolumes", # не изменять
    "registered_volumes_writing": true, # если указать false, то результаты совмещения не будут отписываться, только то, что в консоли
    "minimize_padding" : true,
    "calculate_metrics": false, # флаг для вычисления метрик - для этого нужно указать эталонное преобразование gt по примеру initial (будет ниже)
    "path_to_gt_matrix_json" : "",
    "SelectedVisualizedMetricsList" : ["MSE",
                                       "normalized maximum deviation of distances (from geometry MSE)",
                                       "maximum deviation of distances (from geometry MSE)",
                                       "norm_geometry_rmse",
                                       "geometry_rmse",
                                       "norm_geometry_MSE"],

    "path_to_markup" : "/path/to/experiments/test_data/nut/", # путь до неизменяемого объема, к которому будет приводится объем moving, может иметь любое расположение
    "path_to_moving" : "/path/to/experiments/test_data/nut_rotated/", # путь до изменяемого объема, который будет приводится к markup, может иметь любое расположение
    "path_to_inital_transform_matrix_json" : "/path/to/experiments/test_data/nut_rotated/initial_matrix.json" # можно указать начальное преобразование для помощи методу. Пример json'a с нач. преобразование содержится в папке тестового объема nut_rotated 
}


```

</details>
