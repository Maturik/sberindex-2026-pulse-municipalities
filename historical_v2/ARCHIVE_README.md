# СберИндекс V2

Новая экономическая задача конкурса 2026. Основной отчёт: METHOD_REPORT.md; презентация: presentation.pdf; редактируемый источник: presentation.pptx; локальная страница: index.html. Код/data/config разделены. Исходные данные и веса не включены.

## Запуск

Проверенная среда: Windows, Python3.12, CPU. Linux API не требует TEMP: пути задаются явно до импорта численных библиотек. Сам полный запуск Linux не выполнялся.

1. Получить официальный consumption.parquet в личной рабочей папке из ссылки конкурса. SHA должен совпасть с config/PROTOCOL.json.
2. Создать venv. Установить CPU torch2.8.0: `python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`; затем `python -m pip install -r requirements.txt`.
3. Скачать README.md, config.json и model.safetensors amazon/chronos-2 revision 29ec3766d36d6f73f0696f85560a422f50e8498c в LOCAL_MODEL. SHA каждого файла указан в JSON; runner проверяет его до загрузки. Прогноз рассчитывается локально, данные не передаются в сервис прогнозирования.
4. `python -B code/v2_reproduce.py --data /absolute/path/consumption.parquet --protocol config/PROTOCOL.json --model-dir /absolute/path/LOCAL_MODEL --run-dir /absolute/path/NEW_RUN --temp-dir /absolute/path/WRITABLE_TMP`
5. `python -B /absolute/path/NEW_RUN/code/v2_run.py --run-dir /absolute/path/NEW_RUN --temp-dir /absolute/path/WRITABLE_TMP`
6. После завершения run, из исходного каталога этого пакета: `python -B code/v2_forecast_all.py --run-dir /absolute/path/NEW_RUN --output-dir /absolute/path/NEW_PROJECTION --temp-dir /absolute/path/WRITABLE_TMP`. Скрипт берёт сохранённого победителя, проверяет freeze и требует новый output-dir. Рядом с ним должен лежать неизменный v2_forecasting.py; подготовка не копирует этот отдельный скрипт в NEW_RUN/code. Это архивная проекция января-декабря 2025 из декабря 2024, а не новый тест или прогноз текущего 2026 года.

Windows: указать абсолютные G-пути. Для TEMP желателен ASCII-путь, например G:\Codex\sberindex2026_runtime. `--project-root` нужен только для локальных каталогов зависимостей; обычной venv не нужен. Новый run-dir обязателен. BURNED - защита от повторного запуска после уже раскрытой оценки; код не перезаписывает успешный или неуспешный научный run.

Для проверки искусственных условий: `python -B code/v2_tests.py`. Экономический результат получен однажды по замороженному протоколу; переносимая подготовка восстановила те же входы. Отдельного полного запуска публикуемого пакета с нуля, Linux-запуска и Cloud-повтора нет. Совпадение SHA исходников и агрегатов не заменяет такой повтор.

## Результат

FORECAST_SELECTION.json хранит единственный validation winner до held. DETECTOR_SELECTION.json хранит выбранный детектор до held. ТаблицыMETRICS/CATEGORY_METRICS показывают все методы; h12 - diagnostic. Полные индивидуальные forecast/residual CSV остаются в локальном run, в публичный пакет не включены. `results/RESULT.json` - первичная квитанция расчёта, её старое CLAUDE_REVIEW=NOT_RUN относится к моменту исполнения; finalreview см. provenance после фактической проверки пакета.

## Лицензии и границы

Runtime dependencies и Chronos-2 устанавливаются отдельно; их полные лицензии/notice в notices. Это не лицензия собственного нового кода. Решение участника о лицензировании собственного кода и передаче материалов - перед публикацией. Данные имеют указанную организатором CC BY-SA4.0, одновременно в Положении есть ограничения передачи; их применимость отдельно не снята. Не включать rawdata, модельные веса, приватные регистрации или чужое закрытое ядро в GitHub.

Публикуемый код воспроизводит численные расчёты и проекцию. Исходники оформления графиков/PDF/PPTX в этот комплект не включены; конечные иллюстрации приложены.
