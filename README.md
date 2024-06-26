# AAA ML Курсовой проект

Тематические фото для привлечения внимания.

<img src=https://github.com/pyrogn/aaa-image-enhancement/assets/60060559/95375b2f-fc93-4851-889e-40052e800f14 height=200>
<img src=https://github.com/pyrogn/aaa-image-enhancement/assets/60060559/938d7e83-e212-453e-998c-4a4d790b72be height=200>
<img src=https://github.com/pyrogn/aaa-image-enhancement/assets/60060559/13f9668f-43e7-4069-b3d9-6a9400410ec6 height=200>

Привлек? Отлично, далее идет описание проекта и остальные детали.

## Тема: 10. Автоулучшение фото

Проект реализует автоулучшение фото. Фокус был направлен на фото с **недостаточной экспозицией** в категории "**Недвижимость**". Проект реализован в виде **сервиса** с временем ответа менее 1 секунды. Для оценки качества улучшений был создан ещё один проект [**Dual-Choice**](https://github.com/pyrogn/aaa-dual-choice). В финальное решение не вошли ML модели, но они были использованы в процессе анализа и прототипирования.

## Презентация решения

[**Google Slides**](https://docs.google.com/presentation/d/1CvOWqobmL6lWt8M2SMFbaOgiBde4Gu7QtYTHPpbqZ6E/edit?usp=sharing)

## Команда

Название: Ассоциация Анонимных Аналитиков

- Ермаков Павел @pyrogn
- Иванов Артем @aert14

## Структура

- `src/aaa_image_enhancement/` - устанавливаемый Python проект с запускаемыми fastapi приложениями
- `experiments/` - различный код
- `notebooks/` - ноутбуки с output, если потребуется для демонстрации
- `models/` - модели или fastapi приложения, задекларированные в Dockerfile
- `benchmarks/` - бенчмарки
- `demo/` - простой сервис для демонстрации работы исправления изображений
- `tests/` - тесты
- **[Dual-Choice](https://github.com/pyrogn/aaa-dual-choice) - проект с оценкой субъективного качества фото**

## Запуск

### Демонстрация

- `make demo` или `docker compose --profiles demo up`
- Смотреть Flask приложение на порту 5555. К примеру, http://127.0.0.1:5555/
- Можно вставить множество картинок
- Картинки отправляются на эндпоинт `/enhance_image`. Слева — оригинальная фотография. Если улучшения нет, то правая фотография будет отсутствовать.

<img src=https://github.com/pyrogn/aaa-image-enhancement/assets/60060559/3f712853-6713-4b6e-af8f-3bf15396c1d0 height=400>

### Запуск бэкенда

- `make up` или `docker compose up` для запуска главного приложения, приложения-детектора и приложения-улучшалки
- Взаимодействовать с портом 8000 по эндоинтам, описанным в главном приложении.
- [Код главного приложения с описанием эндпоинтов](./src/aaa_image_enhancement/apps/main.py)
- [Код приложения-детектора](./src/aaa_image_enhancement/apps/detector.py)
- [Код приложения-улучшалки](./src/aaa_image_enhancement/apps/enhancer.py)

### Выключение

- `make down` или `docker compose --profile "*" down`

### Бенчмарки

- Поднять сервис на сервере через `make up`.
- Установить Rye на другом устройстве/сервере. Выполнить `rye sync --features test`. (знаю, что сложно, возможно, подключу devcontainers или запуск через докер)
- `locust -f benchmarks/benchmark_app.py --host=http://51.250.19.218:8000` (вставить адрес сервера, который будем нагружать и запустить нагрузку через интерфейс http://localhost:8089/).

На сервере с двумя 2GHz ядрами достигается ~ 15 RPS с временем ответа 100-300ms.

### Тестирование

- Иметь запущенный Docker Engine
- `make test`
- Контейнеры удалятся по окончанию тестирования

## Инфраструктура

- Rye для менеджмента зависимостей
- FastAPI, uvicorn для бэкенда
- Docker Compose
- pytest внутри Docker для тестирования
- CI для независимого запуска тестов и линтера
- locust для нагрузочного тестирования

## Модели

- [Illumination-Adaptive-Transformer](https://github.com/cuiziteng/Illumination-Adaptive-Transformer)

## Архитектура

Это эксперимент в [mermaid](https://mermaid.js.org/).

### Архитектура системы

```mermaid
graph LR
    subgraph "Internet"
        style Internet fill:transparent, stroke-dasharray: 5 5
        client[Client]
        subgraph "Server (Docker network)"
            main[Main App]
            detector[Detector App]
            enhancer[Enhancer App]
        end
    end

    client <--> main
    main <--> detector
    main <--> enhancer
```

### API

```mermaid
graph LR
    A[Client] -->|Upload Image| B["/detect_problems"]
    B -->|JSON with Defects| A
```
```mermaid
graph LR
    A[Client] -->|Upload Image| C["/enhance_image"]
    C -->|"Enhanced Image or 204 (no enhancement)"| A
```
```mermaid
graph LR
    A[Client] -->|Upload Image and Defect Name| D["/fix_defect"]
    D -->|Enhanced Image| A
```

Красивую автоматическую документацию также можно посмотреть, если
- Поднять сервис `make up`
- Зайти на http://{host}:8000/docs или http://{host}:8000/redoc

<img height="300" alt="image" src="https://github.com/pyrogn/aaa-image-enhancement/assets/60060559/c1f25fd7-0c15-4d4a-8b91-8812482b89cd">

### Code structure

Сложно передать диаграммой, здесь поверхностно и упущена часть взаимодействий и атрибутов, но по сути.

Пояснения:
- `ImageConversions` - вспомогательный класс для манипулирования картинками.
- `ImageDefects` - это датакласс, который создаётся из Enum `DefectsNames` и используется для передачи информации о дефектах.
- `DefectsDetector`  содержит в себе список из функций-детекторов, которые по картинке выдают найденные дефекты, которые будут отображены в возвращаемом `ImageDefects`.
- `ImageEnhancer` имеет мапу из функций (DefectNames => Callable), и для конкретного дефекта вызывает соответсвующую функцию над картинкой для исправления.
- `EnhanceStrategy` нужен для принятия решения об исправлении ряда дефектов. К примеру, `EnhanceStrategyFirst` исправляет только первый (самый важный). И надо соответствовать интерфейсу.

```mermaid
classDiagram
    class DefectsDetector {
        +find_defects(image: ImageConversions) : ImageDefects
    }
    class ImageEnhancer {
        +fix_defect(defect: DefectNames) : np.ndarray
    }
    class EnhanceStrategy {
        <<interface>>
        +enhance_image() : np.ndarray
    }
    class EnhanceStrategyFirst {
        +enhance_image() : np.ndarray
    }
    class ImageDefects {
        +has_defects() : bool
    }
    class DefectNames {
        <<enumeration>>
        BLUR
        LOW_LIGHT
        LOW_CONTRAST
        POOR_WHITE_BALANCE
        NOISY
        ...
    }
    class ImageConversions {
        +to_numpy() : np.ndarray
        +to_pil() : Image.Image
        +to_cv2() : np.ndarray
        +to_grayscale() : np.ndarray
    }

    DefectsDetector --> ImageDefects
    DefectsDetector --> ImageConversions

    ImageEnhancer --> DefectNames

    EnhanceStrategy <|.. EnhanceStrategyFirst

    EnhanceStrategyFirst --> ImageDefects
    EnhanceStrategyFirst --> ImageEnhancer
```

## Работа с репозиторием
[![Rye](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/rye/main/artwork/badge.json)](https://rye-up.com) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

- Относимся к main ветке осторожно, добавляем коммиты через PR. Работаем в своей ветке.
- Используем пакетный менеджер [Rye](https://github.com/astral-sh/rye) (`rye sync --all-features`).
- Не заливаем данные:
  - В jupyter notebook перед отправкой удаляем весь output (Папка `notebooks/` добавлена в исключение, там может быть output).
  - Картинки и гифки не оставляем в репо, а заливаем на хранилище GitHub через вставку через веб-интерфейс.
- Прогоняем код через Ruff (`rye run lint`, source находится в pyproject). Индивидуально: (`rye lint --fix`, `rye fmt`, либо `ruff check --fix`, `ruff format`).
- Проверяем тесты `make test`.
- Все или почти все эти операции можно включить через `pre-commit install`. Можно запустить все проверки через `rye run pre` или `pre-commit run --all-files`.
- Если инструмент работает некорректно, можно добавлять точечно `noqa: <code>`, `type: ignore` или добавить исключения в конфиге в `pyproject.toml`. Или подредактировать `.pre-commit-config.yaml`.
- Можно переносить и переименовывать файлы, функции, переменные. Но только через рефакторинг (как F2 или Refactor... в VSCode), чтобы ничего не сломалось.
- ~~Типизация `mypy ./src`~~

## Выводы

- Проект интересный, с точки зрения ML части слабоватый. С точки зрения анализа пробемы хороший.
- Тесты неплохие, но по-хорошему надо добавить тесты на fastapi с моками, так будет правильнее, чем тестировать на реальном сервисе через docker.
