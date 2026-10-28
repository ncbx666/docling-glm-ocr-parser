# Гибридный PDF-парсер: Docling + GLM-OCR

Локальный парсер PDF в Markdown. Основной парсер — Docling, а GLM-OCR
получает только сложные визуальные фрагменты.

- **Docling** обрабатывает обычный текст, структуру страниц, порядок чтения,
  таблицы и изображения.
- **GLM-OCR** (`zai-org/GLM-OCR`) вызывается для найденных Docling таблиц,
  рисунков, графиков и схем, формул, рукописных и анкетных областей, а также
  для страниц, где почти нет извлекаемого текста.
- Промежуточные результаты (экспорт Docling, изображения страниц, вырезанные
  области, таблицы в CSV/HTML, список целей и ответы GLM-OCR) сохраняются
  рядом с результатом.

Тот же подход, но с MinerU вместо GLM-OCR:
[docling-miner-pdf-parser](https://github.com/ncbx666/docling-miner-pdf-parser).

## Установка (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

По умолчанию GLM-OCR запускается через официальный путь Transformers на CPU.
При первом запуске модель скачивается с Hugging Face: это долго, нужно
достаточно места на диске и RAM. На CPU распознавание работает медленно.

Если pip не подобрал CPU-сборку PyTorch, поставьте её из официального индекса
и переустановите зависимости:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Запуск

Один PDF в Markdown-файл:

```powershell
python parse_pdf.py "C:\path\input.pdf" --output "C:\path\result.md"
```

Один PDF в папку:

```powershell
python parse_pdf.py "C:\path\input.pdf" --output "C:\path\results"
```

Папка с PDF в папку с Markdown:

```powershell
python parse_pdf.py "C:\path\input_folder" --output "C:\path\result_folder" --continue-on-error
```

### Бэкенд Ollama

Если локально установлен Ollama и скачана официальная модель `glm-ocr`:

```powershell
python parse_pdf.py "C:\path\input.pdf" --output "C:\path\result.md" --glm-backend ollama
```

## Параметры

| Флаг | По умолчанию | Что делает |
|---|---|---|
| `--output` | — | Markdown-файл для одного PDF или папка для результатов |
| `--glm-backend` | `transformers` | `transformers`, `ollama` или `none` (только Docling, для отладки) |
| `--glm-model` | `zai-org/GLM-OCR` | id модели на Hugging Face для бэкенда transformers |
| `--glm-max-new-tokens` | `8192` | максимум токенов ответа GLM-OCR на одну область |
| `--images-scale` | `2.0` | масштаб рендеринга страниц и областей: выше — точнее OCR, но больше памяти и времени |
| `--low-text-threshold` | `80` | страница с меньшим числом букв и цифр целиком уходит в GLM-OCR |
| `--continue-on-glm-error` | выкл. | при ошибке GLM-OCR оставить результат Docling и пометить область |
| `--continue-on-error` | выкл. | для папки: не останавливаться на упавшем PDF, сводка ошибок в конце |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

`--glm-backend none` нужен только для отладки Docling: полноценного гибридного
OCR в этом режиме нет.

## Результаты

Для `result.md` артефакты сохраняются в `result_artifacts`:

- `result_parse_pdf.log` — лог запуска;
- `docling.md`, `docling.txt`, `docling.json` — экспорт только Docling;
- `pages\` — изображения страниц и текст/Markdown Docling по страницам;
- `regions\` — области, отправленные в GLM-OCR;
- `tables\` — таблицы Docling в CSV/HTML;
- `targets.json` — выбранные для GLM-OCR области и причины выбора;
- `glm_ocr_results.json` — ответы GLM-OCR по каждой области;
- `manifest.json` — настройки, тайминги, список целей и пути к файлам.

Итоговый Markdown содержит комментарии с общим временем работы и текст Docling
по страницам. Результаты GLM-OCR вставляются после содержимого своей страницы
в порядке чтения.

Для папки лог пишется в `result_folder\parse_pdf.log`, Markdown повторяет
структуру входной папки, а артефакты лежат в `result_folder\_artifacts`.

## Особенности

- GLM-OCR не прогоняет каждую страницу. Страница целиком уходит в GLM-OCR,
  только если Docling извлёк меньше `--low-text-threshold` букв и цифр.
- Сначала всегда работает разметка Docling, области для GLM-OCR вырезаются из
  его изображений страниц и элементов.
- Пути обрабатываются через `pathlib`, поэтому Windows-пути работают.

## Лицензия

[MIT](LICENSE)
