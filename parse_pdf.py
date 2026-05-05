from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from hybrid_pdf_parser.models import ParseSettings
from hybrid_pdf_parser.pipeline import HybridPdfPipeline

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        input_path = Path(args.input).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()
        jobs = plan_jobs(input_path, output_path)
        attach_file_logger(log_file_path(input_path, output_path, jobs), args.log_level)
        settings = ParseSettings(
            images_scale=args.images_scale,
            low_text_threshold=args.low_text_threshold,
            glm_max_new_tokens=args.glm_max_new_tokens,
            glm_model=args.glm_model,
            glm_backend=args.glm_backend,
            continue_on_glm_error=args.continue_on_glm_error,
        )
        pipeline = HybridPdfPipeline(settings)

        started = time.perf_counter()
        failures: list[tuple[Path, str]] = []
        for pdf_path, markdown_path, artifacts_dir in jobs:
            try:
                LOGGER.info("Processing %s", pdf_path)
                pipeline.process_pdf(pdf_path, markdown_path, artifacts_dir)
                LOGGER.info("Wrote %s", markdown_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed processing %s", pdf_path)
                failures.append((pdf_path, str(exc)))
                if not args.continue_on_error:
                    raise

        elapsed = time.perf_counter() - started
        LOGGER.info("Whole pipeline finished in %.2f seconds", elapsed)
        if failures:
            LOGGER.error("%s file(s) failed:", len(failures))
            for pdf_path, error in failures:
                LOGGER.error("  %s: %s", pdf_path, error)
            return 2
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("%s", exc)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid local PDF parser: Docling for structure/text plus GLM-OCR for complex visual regions.",
    )
    parser.add_argument("input", help="Input PDF file or folder containing PDFs.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output Markdown file for one PDF, or output folder for an input folder.",
    )
    parser.add_argument(
        "--glm-backend",
        choices=["transformers", "ollama", "none"],
        default="transformers",
        help="Replaceable GLM-OCR backend. Use 'none' only for Docling-only debugging.",
    )
    parser.add_argument(
        "--glm-model",
        default="zai-org/GLM-OCR",
        help="Hugging Face model id for the transformers backend.",
    )
    parser.add_argument(
        "--glm-max-new-tokens",
        type=int,
        default=8192,
        help="Maximum GLM-OCR output tokens per target region.",
    )
    parser.add_argument(
        "--images-scale",
        type=float,
        default=2.0,
        help="Docling page/region image render scale. Higher improves OCR crops but costs memory/time.",
    )
    parser.add_argument(
        "--low-text-threshold",
        type=int,
        default=80,
        help="Alphanumeric character threshold below which a page is sent to GLM-OCR as scanned/low-text.",
    )
    parser.add_argument(
        "--continue-on-glm-error",
        action="store_true",
        help="Keep Docling output and mark failed GLM-OCR regions instead of aborting.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="For folder input, continue after a PDF fails and report failures at the end.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level.",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def attach_file_logger(path: Path, level: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logging.getLogger().addHandler(handler)
    LOGGER.info("Writing run log to %s", path)


def log_file_path(
    input_path: Path,
    output_path: Path,
    jobs: list[tuple[Path, Path, Path]],
) -> Path:
    if input_path.is_dir():
        return output_path / "parse_pdf.log"
    return jobs[0][1].parent / f"{jobs[0][1].stem}_parse_pdf.log"


def plan_jobs(input_path: Path, output_path: Path) -> list[tuple[Path, Path, Path]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file must be a PDF: {input_path}")
        markdown_path = _single_output_path(input_path, output_path)
        artifacts_dir = markdown_path.parent / f"{markdown_path.stem}_artifacts"
        return [(input_path, markdown_path, artifacts_dir)]

    if not input_path.is_dir():
        raise ValueError(f"Input path is neither a file nor a folder: {input_path}")
    if output_path.suffix.lower() == ".md":
        raise ValueError("Folder input requires --output to be a folder, not a .md file.")

    pdfs = sorted(path for path in input_path.rglob("*.pdf") if path.is_file())
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found under: {input_path}")

    jobs: list[tuple[Path, Path, Path]] = []
    for pdf_path in pdfs:
        relative = pdf_path.relative_to(input_path)
        markdown_path = (output_path / relative).with_suffix(".md")
        artifact_name = "__".join(relative.with_suffix("").parts) + "_artifacts"
        artifacts_dir = output_path / "_artifacts" / artifact_name
        jobs.append((pdf_path, markdown_path, artifacts_dir))
    return jobs


def _single_output_path(input_pdf: Path, output_path: Path) -> Path:
    if output_path.suffix.lower() == ".md":
        return output_path
    if output_path.exists() and output_path.is_dir():
        return output_path / f"{input_pdf.stem}.md"
    if output_path.suffix == "":
        return output_path / f"{input_pdf.stem}.md"
    raise ValueError(
        "For a single PDF, --output must be either a .md file or a folder path."
    )


if __name__ == "__main__":
    sys.exit(main())
