"""Run the installed wheel with an explicitly sealed AWQ source stage."""

import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    stage = Path(os.environ["GPU_CACHE_RUNTIME_STAGE"]).resolve()
    receipt = json.loads((stage / "identity.json").read_text())
    for name, expected in receipt["overrides"].items():
        path = stage / "vllm" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("AWQ runtime stage changed: " + name)
    sys.path.insert(0, str(stage))
    import vllm

    if Path(vllm.__file__).absolute().parent != stage / "vllm":
        raise RuntimeError("AWQ runtime did not resolve to the declared stage")
    print("GPU_CACHE_AWQ_STAGE " + json.dumps(dict(stage=str(stage), identity=receipt,
          chunk_rows=int(os.environ.get("VLLM_AWQ_REPACK_CHUNK_ROWS", "0")))), flush=True)
    from vllm.entrypoints.cli.main import main as serve

    serve()


if __name__ == "__main__":
    main()
