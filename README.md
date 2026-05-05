# Hybrid PDF Parser: Docling + GLM-OCR

This project parses PDFs locally on Windows 11 with Docling as the main parser, then sends only complex visual targets to GLM-OCR:

- Docling handles normal text, page structure, reading order, tables, and images.
- GLM-OCR is called for Docling-detected tables, pictures/figures, charts/diagrams, formulas, handwriting/form-like regions, and pages with very low extracted text.
- Intermediate Docling exports, page images, region crops, table CSV/HTML files, target manifests, and GLM-OCR results are saved next to each output.

## Setup in PowerShell

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default backend uses the official GLM-OCR Transformers path with `zai-org/GLM-OCR` on CPU. The first run downloads model files from Hugging Face, so it can take a long time and needs enough disk/RAM. CPU inference is expected to be slow.

If PyTorch CPU wheels are not selected automatically on your machine, install PyTorch from the official CPU index first, then reinstall the requirements:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Run

Single PDF to a Markdown file:

```powershell
python parse_pdf.py "C:\path\input.pdf" --output "C:\path\result.md"
```

Single PDF to an output folder:

```powershell
python parse_pdf.py "C:\path\input.pdf" --output "C:\path\results"
```

Folder of PDFs to a folder of Markdown files:

```powershell
python parse_pdf.py "C:\path\input_folder" --output "C:\path\result_folder" --continue-on-error
```

Useful debug options:

```powershell
python parse_pdf.py "C:\path\input.pdf" --output "C:\path\result.md" --log-level DEBUG
python parse_pdf.py "C:\path\input.pdf" --output "C:\path\result.md" --glm-backend none
python parse_pdf.py "C:\path\input.pdf" --output "C:\path\result.md" --continue-on-glm-error
```

`--glm-backend none` is only for debugging Docling extraction and artifact generation. It does not satisfy the full hybrid OCR goal.

## Optional Ollama backend

If you have Ollama installed locally and have pulled the official `glm-ocr` model, you can use:

```powershell
python parse_pdf.py "C:\path\input.pdf" --output "C:\path\result.md" --glm-backend ollama
```

## Outputs

For `result.md`, artifacts are saved in `result_artifacts`:

- `result_parse_pdf.log`: run log for the single-file command.
- `docling.md`, `docling.txt`, `docling.json`: Docling-only exports.
- `pages\`: rendered page images and per-page Docling text/Markdown.
- `regions\`: Docling crops sent to GLM-OCR.
- `tables\`: Docling table CSV/HTML exports where available.
- `targets.json`: selected GLM-OCR targets and reasons.
- `glm_ocr_results.json`: GLM-OCR output for each target.
- `manifest.json`: settings, timings, target list, and output paths.

The generated Markdown includes total pipeline timing comments and page-ordered Docling Markdown. GLM-OCR sections are inserted after the corresponding page content in Docling reading-order target order.

For folder input, the run log is `result_folder\parse_pdf.log`, Markdown outputs mirror the input folder structure, and per-document artifacts are stored under `result_folder\_artifacts`.

## Notes

- The code intentionally does not parse every page with GLM-OCR. Full-page GLM-OCR is used only when Docling extracts fewer than `--low-text-threshold` alphanumeric characters from a page.
- Docling table/image/layout detection is always used first. GLM-OCR target crops are derived from Docling page and element images.
- All paths are handled with Python `pathlib` and work with Windows paths.
