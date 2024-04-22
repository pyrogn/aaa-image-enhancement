# AAA ML Курсовой проект

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
- Прогоняем код через Ruff и Black (`rye lint --fix`, `rye fmt`, либо `ruff check --fix`, `black .`).
- Проверяем тесты `rye test` или `pytest`.
- Возможно, добавлю тесты, форматирование и линт в `pre-commit`, чтобы делать меньше движений.
- Если инструмент работает некорректно, можно добавлять точечно `noqa: <code>`, `type: ignore` или добавить исключения в конфиге в `pyproject.toml`.
- ~~Типизация `mypy ./src`~~
- More to come...