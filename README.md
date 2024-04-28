# AAA ML Курсовой проект

<img src=https://github.com/pyrogn/aaa-image-enhancement/assets/60060559/95375b2f-fc93-4851-889e-40052e800f14 height=200>
<img src=https://github.com/pyrogn/aaa-image-enhancement/assets/60060559/938d7e83-e212-453e-998c-4a4d790b72be height=200>
<img src=https://github.com/pyrogn/aaa-image-enhancement/assets/60060559/13f9668f-43e7-4069-b3d9-6a9400410ec6 height=200>


## Тема: 10. Автоулучшение фото

> Проект направлен на создание алгоритма, который повысит визуальное восприятие изображений недвижимости, загружаемых пользователями. Основная цель - улучшить такие параметры фото, как контрастность, насыщенность, яркость, шум, тени и др. Для этого планируется протестировать существующие open-source решения, дообучить их или разработать собственный алгоритм, основанный на нейронных сетях или классическом компьютерном зрении. Качество решения будет провалидировано с учетом субъективности визуальных улучшений. Финальный продукт будет реализован в виде микросервиса с временем ответа не более 1 секунды на фото и, возможно, с опцией ручной корректировки параметров.

## Команда

Название: Ассоциация Анонимных Аналитиков

- Ермаков Павел @pyrogn
- Иванов Артем @aert14

## Работа с репозиторием
[![Rye](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/rye/main/artwork/badge.json)](https://rye-up.com) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

- Относимся к main ветке осторожно, добавляем коммиты через PR. Работаем в своей ветке.
- Используем пакетный менеджер [Rye](https://github.com/astral-sh/rye) (`rye sync --all-features`). 
- Не заливаем данные:
  - В jupyter notebook перед отправкой удаляем весь output.
  - Картинки и гифки не оставляем в репо, а заливаем на хранилище GitHub через вставку через веб-интерфейс.
- Прогоняем код через Ruff (`rye lint --fix`, `rye fmt`, либо `ruff check --fix`, `ruff format`).
- Проверяем тесты `rye test` или `pytest`.
- Возможно, добавлю тесты, форматирование и линт в `pre-commit`, чтобы делать меньше движений.
- Если инструмент работает некорректно, можно добавлять точечно `noqa: <code>`, `type: ignore` или добавить исключения в конфиге в `pyproject.toml`.
- Можно переносить и переименовывать файлы, функции, переменные. Но только через рефакторинг (как F2 или Refactor... в VSCode), чтобы ничего не сломалось.
- ~~Типизация `mypy ./src`~~
- More to come...

## Инфраструктура

- https://pytorch.org/serve/

```mermaid
flowchart TB
    subgraph docker1["Docker Container 1"]
        torchServe1[TorchServe]
        model1[PyTorch Model 1]
        torchServe1 --> model1
    end

    subgraph docker2["Docker Container 2"]
        torchServe2[TorchServe]
        model2[PyTorch Model 2]
        torchServe2 --> model2
    end

    subgraph docker3["Docker Container 3"]
        torchServe3[TorchServe]
        model3[PyTorch Model 3]
        torchServe3 --> model3
    end

    subgraph dockermain["Docker Container for Service"]
        mainModule[Main Module]
    end
        mainModule -->|Sends image| docker1
        mainModule -->|Sends image| docker2
        mainModule -->|Sends image| docker3

    docker1 -->|Returns new image| mainModule
    docker2 -->|Returns new image| mainModule
    docker3 -->|Returns new image| mainModule

    style mainModule fill:#f9f,stroke:#333,stroke-width:2px
    style docker1 fill:#bbf,stroke:#333,stroke-width:2px
    style docker2 fill:#bbf,stroke:#333,stroke-width:2px
    style docker3 fill:#bbf,stroke:#333,stroke-width:2px
```

## Модели

- model1 (github link, citation)

## Архитектура

Это эксперимент в [mermaid](https://mermaid.js.org/).

### Autonomous

```mermaid
graph LR
    Image -->|input| DefectsDetector
    Image -->|input| Enhancer
    DefectsDetector -->|returns image defects| Enhancer
    Enhancer -->|produces enhanced image| EnhancedImage
```

```mermaid
graph LR
    subgraph Service
        DefectsDetector[Defects Detector]
        Enhancer[Enhancer]
    end

    User[User] -->|uploads image| Image[Image]
    Image -->|input| DefectsDetector
    DefectsDetector -->|returns image defects| Enhancer
    Enhancer -->|produces enhanced image| EnhancedImage[Enhanced Image]

    User -->|specifies defect and uploads image| DirectEnhancement[Image & Specific Defect]
    DirectEnhancement -->|input| Enhancer
```

### Interactive with User

```mermaid
sequenceDiagram
    participant User
    participant DefectsDetector
    participant Enhancer

    box Client
    participant User
    end 

    box Server
    participant DefectsDetector
    participant Enhancer
    end 

    User ->>+ DefectsDetector: Upload Image
    DefectsDetector -->>- User: Return Image Defects

    alt User Chooses to Take New Picture
        User ->> User: Take New Picture
    else User Chooses to Enhance Image
        User ->>+ Enhancer: Send Image and Returned Defects
        Enhancer -->>- User: Return Enhanced Image

        alt User Keeps Enhanced Image
            User ->> User: Keep Enhanced Image
        else User Discards Enhanced Image
            User ->> User: Discard Enhanced Image
        end
    end
```
