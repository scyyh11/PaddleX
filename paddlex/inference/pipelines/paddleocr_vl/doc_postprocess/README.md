# PaddleOCR-VL Multi-Page Document Post-Processing

Document-level post-processing for multi-page `PaddleOCR-VL-1.8` results: recovers global heading hierarchies, joins cross-page tables and paragraphs, repairs sparse table columns and formula delimiters, and renders a unified Markdown document.

## Usage in `PaddleOCR-VL-1.8`

Configured in `paddlex/configs/pipelines/PaddleOCR-VL-1.8.yaml`:

- `use_doc_postprocess: null` — auto-enabled when a valid model bundle is found; falls back to basic PaddleX restructuring with a one-time warning if missing. Set `true` to require models or `false` to disable.
- `doc_postprocess_models_dir: null` — explicit bundle directory (overrides `$PADDLEX_DOC_POSTPROCESS_MODELS` / `$PADDLEOCR_DOC_POSTPROCESS_MODELS` / `~/.paddleocr/doc_postprocess/models`).
- `doc_postprocess_device: "cpu"` — `"cpu"` (default), `"auto"`, `"gpu"`, or `"gpu:<index>"`.

```python
from paddlex import create_pipeline

pipeline = create_pipeline("PaddleOCR-VL-1.8")
pages = list(pipeline.predict("report.pdf"))
doc = next(pipeline.restructure_pages(pages, concatenate_pages=True))
doc.save_to_markdown("output")
```

Standalone usage:

```python
from paddlex.inference.pipelines.paddleocr_vl.doc_postprocess import (
    DocPostprocessConfig,
    DocPostprocessor,
)

processor = DocPostprocessor(models_dir="/path/to/models", device="cpu")
doc = processor.process(pages)
```

## Portable Model Bundle (`.safetensors`)

To avoid `pickle`/`joblib` security and library-version coupling, convert an existing `.joblib` bundle into a portable `.safetensors` + `.json` bundle:

```bash
python -m paddlex.inference.pipelines.paddleocr_vl.doc_postprocess.bundle export-portable /path/to/joblib-models /path/to/portable-models
python -m paddlex.inference.pipelines.paddleocr_vl.doc_postprocess.bundle check /path/to/portable-models
```

## Architecture

- `config.py` — frozen `DocPostprocessConfig` dataclass for all pipeline thresholds and rule switches.
- `features.py` — multi-threaded grayscale conversion (libpng-exact) and Otsu-binarized typography/line measurements.
- `portable_trees.py` / `models.py` / `bundle.py` — safe `.safetensors` tree serialization, pure-NumPy evaluator, and instance-scoped `ModelStore`.
- `pipeline.py` — `DocPostprocessor` orchestrator with request-local `ContextVar` scoping.
- `result.py` — `DocPostprocessResult` compatible with PaddleX Markdown/JSON output conventions.
- `engine/` — 5-stage document engine (`headings` -> `tables` -> `text` -> `graphic/layout cleanup` -> `render`).
