from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from .models import DocumentArtifacts, DoclingParseResult, ParseSettings, TargetRegion

LOGGER = logging.getLogger(__name__)


class DoclingAdapter:
    """Docling conversion and artifact extraction using documented public APIs."""

    def __init__(self, settings: ParseSettings) -> None:
        self.settings = settings
        self._converter = None
        self._doc_classes: dict[str, Any] = {}

    def parse(self, pdf_path: Path, artifacts_root: Path) -> DoclingParseResult:
        pdf_path = pdf_path.resolve()
        artifacts = _prepare_artifacts(artifacts_root)
        converter = self._get_converter()

        LOGGER.info("Converting with Docling: %s", pdf_path)
        started = time.perf_counter()
        conv_res = converter.convert(pdf_path)
        docling_seconds = time.perf_counter() - started
        doc = conv_res.document

        self._save_docling_exports(doc, artifacts)
        page_image_paths = self._save_page_images(doc, artifacts)
        page_markdown = self._export_page_markdown(doc, artifacts)
        page_text = self._export_page_text(doc, artifacts)
        targets = self._collect_targets(doc, artifacts, page_image_paths, page_text)

        return DoclingParseResult(
            pdf_path=pdf_path,
            artifacts=artifacts,
            page_markdown=page_markdown,
            page_text=page_text,
            targets=targets,
            docling_seconds=docling_seconds,
            page_count=len(page_markdown),
        )

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling_core.types.doc import DocItemLabel, PictureItem, TableItem
        except ImportError as exc:
            raise RuntimeError(
                "Docling is not installed or cannot be imported. Install dependencies "
                "with: python -m pip install -r requirements.txt"
            ) from exc

        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = self.settings.images_scale
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        if hasattr(pipeline_options, "generate_table_images"):
            pipeline_options.generate_table_images = True
        if hasattr(pipeline_options, "do_table_structure"):
            pipeline_options.do_table_structure = True
        if hasattr(pipeline_options, "do_ocr"):
            pipeline_options.do_ocr = True

        self._doc_classes = {
            "DocItemLabel": DocItemLabel,
            "PictureItem": PictureItem,
            "TableItem": TableItem,
        }
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        return self._converter

    def _save_docling_exports(self, doc: Any, artifacts: DocumentArtifacts) -> None:
        markdown = doc.export_to_markdown(
            page_break_placeholder="\n\n<!-- page-break -->\n\n",
            traverse_pictures=True,
        )
        (artifacts.root / "docling.md").write_text(markdown, encoding="utf-8")
        text = doc.export_to_text(
            page_break_placeholder="\n\n--- page-break ---\n\n",
            traverse_pictures=True,
        )
        (artifacts.root / "docling.txt").write_text(text, encoding="utf-8")

        json_path = artifacts.root / "docling.json"
        try:
            if hasattr(doc, "save_as_json"):
                doc.save_as_json(json_path)
            elif hasattr(doc, "model_dump_json"):
                json_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
            else:
                json_path.write_text(json.dumps({"repr": repr(doc)}, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not save Docling JSON artifact: %s", exc)

    def _save_page_images(self, doc: Any, artifacts: DocumentArtifacts) -> dict[int, Path]:
        page_image_paths: dict[int, Path] = {}
        for page_no, page in sorted(doc.pages.items()):
            real_page_no = int(getattr(page, "page_no", page_no))
            image_ref = getattr(page, "image", None)
            pil_image = getattr(image_ref, "pil_image", None) if image_ref else None
            if pil_image is None:
                LOGGER.warning("No rendered page image available for page %s", real_page_no)
                continue
            path = artifacts.pages_dir / f"page_{real_page_no:04d}.png"
            pil_image.save(path, format="PNG")
            page_image_paths[real_page_no] = path
        return page_image_paths

    def _export_page_markdown(self, doc: Any, artifacts: DocumentArtifacts) -> dict[int, str]:
        page_markdown: dict[int, str] = {}
        for page_no in sorted(int(p) for p in doc.pages.keys()):
            markdown = doc.export_to_markdown(page_no=page_no, traverse_pictures=True)
            page_markdown[page_no] = markdown.strip()
            (artifacts.pages_dir / f"page_{page_no:04d}_docling.md").write_text(
                markdown,
                encoding="utf-8",
            )
        return page_markdown

    def _export_page_text(self, doc: Any, artifacts: DocumentArtifacts) -> dict[int, str]:
        page_text: dict[int, str] = {}
        for page_no in sorted(int(p) for p in doc.pages.keys()):
            text = doc.export_to_text(page_no=page_no, traverse_pictures=True)
            page_text[page_no] = text.strip()
            (artifacts.pages_dir / f"page_{page_no:04d}_docling.txt").write_text(
                text,
                encoding="utf-8",
            )
        return page_text

    def _collect_targets(
        self,
        doc: Any,
        artifacts: DocumentArtifacts,
        page_image_paths: dict[int, Path],
        page_text: dict[int, str],
    ) -> list[TargetRegion]:
        targets: list[TargetRegion] = []
        target_labels = self._target_labels()
        TableItem = self._doc_classes["TableItem"]
        PictureItem = self._doc_classes["PictureItem"]

        region_index = 0
        for order, (element, _level) in enumerate(doc.iterate_items()):
            page_no = _element_page_no(element)
            if page_no is None:
                continue
            label = _label_value(getattr(element, "label", None))
            is_table = isinstance(element, TableItem) or label == "table"
            is_picture = isinstance(element, PictureItem) or label == "picture"
            if not (is_table or is_picture or label in target_labels):
                continue

            image = _get_element_image(element, doc)
            if image is None:
                LOGGER.warning("Skipping %s on page %s: Docling returned no crop image", label, page_no)
                continue

            kind = _kind_for_label(label)
            region_index += 1
            region_id = f"p{page_no:04d}_{region_index:04d}_{kind}"
            image_path = artifacts.regions_dir / f"{region_id}.png"
            image.save(image_path, format="PNG")

            metadata = self._save_table_artifacts(element, doc, artifacts, region_id) if is_table else {}
            targets.append(
                TargetRegion(
                    region_id=region_id,
                    page_no=page_no,
                    order=order,
                    kind=kind,
                    reason=f"Docling detected {label or kind}",
                    image_path=image_path,
                    docling_ref=getattr(element, "self_ref", None),
                    label=label,
                    metadata=metadata,
                )
            )

        for page_no, text in sorted(page_text.items()):
            text_score = _alnum_count(text)
            page_image = page_image_paths.get(page_no)
            if text_score >= self.settings.low_text_threshold or page_image is None:
                continue
            targets.append(
                TargetRegion(
                    region_id=f"p{page_no:04d}_full_page",
                    page_no=page_no,
                    order=-1,
                    kind="page",
                    reason=(
                        "Low text density detected by Docling "
                        f"({text_score} alphanumeric characters)"
                    ),
                    image_path=page_image,
                    label="page",
                    metadata={"docling_alnum_count": text_score},
                )
            )
        targets.sort(key=lambda item: (item.page_no, item.order, item.region_id))
        _write_json(
            artifacts.root / "targets.json",
            [target.to_json() for target in targets],
        )
        LOGGER.info("Selected %s GLM-OCR target(s)", len(targets))
        return targets

    def _target_labels(self) -> set[str]:
        DocItemLabel = self._doc_classes["DocItemLabel"]
        labels = {
            "table",
            "picture",
            "chart",
            "formula",
            "handwritten_text",
            "form",
        }
        for attr in ("TABLE", "PICTURE", "CHART", "FORMULA", "HANDWRITTEN_TEXT", "FORM"):
            value = getattr(DocItemLabel, attr, None)
            if value is not None:
                labels.add(_label_value(value))
        return labels

    def _save_table_artifacts(
        self,
        element: Any,
        doc: Any,
        artifacts: DocumentArtifacts,
        region_id: str,
    ) -> dict[str, str]:
        metadata: dict[str, str] = {}
        try:
            dataframe = element.export_to_dataframe(doc=doc)
            csv_path = artifacts.tables_dir / f"{region_id}.csv"
            dataframe.to_csv(csv_path, index=False)
            metadata["docling_table_csv"] = str(csv_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not export Docling table CSV for %s: %s", region_id, exc)
        try:
            html_path = artifacts.tables_dir / f"{region_id}.html"
            html_path.write_text(element.export_to_html(doc=doc), encoding="utf-8")
            metadata["docling_table_html"] = str(html_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not export Docling table HTML for %s: %s", region_id, exc)
        return metadata


def _prepare_artifacts(root: Path) -> DocumentArtifacts:
    root.mkdir(parents=True, exist_ok=True)
    pages_dir = root / "pages"
    regions_dir = root / "regions"
    tables_dir = root / "tables"
    pages_dir.mkdir(exist_ok=True)
    regions_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    return DocumentArtifacts(root=root, pages_dir=pages_dir, regions_dir=regions_dir, tables_dir=tables_dir)


def _element_page_no(element: Any) -> int | None:
    prov = getattr(element, "prov", None) or []
    if not prov:
        return None
    page_no = getattr(prov[0], "page_no", None)
    return int(page_no) if page_no is not None else None


def _label_value(label: Any) -> str:
    if label is None:
        return ""
    value = getattr(label, "value", label)
    return str(value)


def _kind_for_label(label: str) -> str:
    if label == "table":
        return "table"
    if label == "formula":
        return "formula"
    if label == "chart":
        return "diagram"
    if label in {"picture", "form", "handwritten_text"}:
        return "image"
    return "text"


def _get_element_image(element: Any, doc: Any) -> Any | None:
    if not hasattr(element, "get_image"):
        return None
    try:
        return element.get_image(doc)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Docling get_image failed for %s: %s", getattr(element, "self_ref", ""), exc)
        return None


def _alnum_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]", text or ""))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

