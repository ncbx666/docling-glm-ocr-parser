from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .docling_adapter import DoclingAdapter
from .glm_ocr import GlmOcrBackend, build_glm_backend
from .models import DoclingParseResult, OcrResult, ParseSettings, TargetRegion

LOGGER = logging.getLogger(__name__)


class HybridPdfPipeline:
    def __init__(self, settings: ParseSettings) -> None:
        self.settings = settings
        self.docling = DoclingAdapter(settings)
        self.glm_backend: GlmOcrBackend = build_glm_backend(settings)

    def process_pdf(self, pdf_path: Path, output_md: Path, artifacts_dir: Path) -> None:
        started = time.perf_counter()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        parse_result = self.docling.parse(pdf_path, artifacts_dir)
        ocr_results = self._run_glm_ocr(parse_result)
        total_seconds = time.perf_counter() - started

        markdown = self._render_markdown(parse_result, ocr_results, total_seconds)
        output_md.write_text(markdown, encoding="utf-8")
        self._write_manifest(parse_result, ocr_results, output_md, total_seconds)
        LOGGER.info("Finished %s in %.2f seconds", pdf_path, total_seconds)

    def _run_glm_ocr(self, parse_result: DoclingParseResult) -> dict[str, OcrResult]:
        results: dict[str, OcrResult] = {}
        for index, target in enumerate(parse_result.targets, start=1):
            LOGGER.info(
                "GLM-OCR target %s/%s: %s (%s)",
                index,
                len(parse_result.targets),
                target.region_id,
                target.reason,
            )
            result = self.glm_backend.recognize(target)
            results[target.region_id] = result
            if result.error and not self.settings.continue_on_glm_error:
                raise RuntimeError(
                    f"GLM-OCR failed for {target.region_id}: {result.error}. "
                    "Use --continue-on-glm-error to keep Docling output and mark failed regions."
                )
        _write_json(
            parse_result.artifacts.root / "glm_ocr_results.json",
            [result.to_json() for result in results.values()],
        )
        return results

    def _render_markdown(
        self,
        parse_result: DoclingParseResult,
        ocr_results: dict[str, OcrResult],
        total_seconds: float,
    ) -> str:
        targets_by_page: dict[int, list[TargetRegion]] = {}
        for target in parse_result.targets:
            targets_by_page.setdefault(target.page_no, []).append(target)

        parts = [
            f"<!-- source: {parse_result.pdf_path} -->",
            f"<!-- pipeline_time_seconds: {total_seconds:.2f} -->",
            f"<!-- docling_time_seconds: {parse_result.docling_seconds:.2f} -->",
            "",
        ]
        for page_no in sorted(parse_result.page_markdown):
            parts.append(f"<!-- Page {page_no} -->")
            base_markdown = parse_result.page_markdown[page_no]
            parts.append(base_markdown if base_markdown else "_No Docling text extracted on this page._")

            for target in targets_by_page.get(page_no, []):
                result = ocr_results.get(target.region_id)
                parts.append(self._render_target(target, result))
            parts.append("")
        return "\n\n".join(part for part in parts if part is not None).rstrip() + "\n"

    def _render_target(self, target: TargetRegion, result: OcrResult | None) -> str:
        title = f"GLM-OCR {target.kind}: {target.reason}"
        image_rel = _safe_markdown_path(target.image_path)
        if result is None:
            body = "_No GLM-OCR result was produced._"
        elif result.error:
            body = f"_GLM-OCR error: {result.error}_"
        elif result.text:
            body = result.text
        else:
            body = "_GLM-OCR returned empty text._"
        return (
            f"<!-- {target.region_id} page={target.page_no} kind={target.kind} -->\n\n"
            f"### {title}\n\n"
            f"Artifact: `{image_rel}`\n\n"
            f"{body}"
        )

    def _write_manifest(
        self,
        parse_result: DoclingParseResult,
        ocr_results: dict[str, OcrResult],
        output_md: Path,
        total_seconds: float,
    ) -> None:
        manifest = {
            "source_pdf": str(parse_result.pdf_path),
            "output_markdown": str(output_md),
            "page_count": parse_result.page_count,
            "docling_seconds": parse_result.docling_seconds,
            "total_seconds": total_seconds,
            "settings": self.settings.__dict__,
            "targets": [target.to_json() for target in parse_result.targets],
            "glm_ocr_results": [result.to_json() for result in ocr_results.values()],
        }
        _write_json(parse_result.artifacts.root / "manifest.json", manifest)


def _safe_markdown_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

