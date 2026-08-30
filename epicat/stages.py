"""The pipeline's stage names, factored out so the CLI can build its argument
parser (and route `--bootstrap`) without importing numpy-dependent modules."""
from __future__ import annotations

STAGES = ("analyse", "render", "captions", "concat", "text", "translate",
          "dub", "mux")
