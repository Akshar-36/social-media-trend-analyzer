"""Remove non-runtime packaging files without changing models or Python source."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def slim_runtime(root, strip_command=None):
    root = Path(root).resolve(strict=True)
    removed = []
    # Keep runtime helpers named "testing" (for example torch.testing).
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        if path.is_dir() and path.name in {"__pycache__", "tests", "test"}:
            shutil.rmtree(path)
            removed.append(str(path.relative_to(root)))

    # These headers support building extensions, which the demo never does.
    for relative in ("torch/include", "pyarrow/include", "numpy/_core/include"):
        path = root / relative
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            removed.append(relative)

    stripped = 0
    if strip_command:
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file() or ".so" not in path.name:
                continue
            with path.open("rb") as handle:
                is_elf = handle.read(4) == b"\x7fELF"
            if is_elf:
                # Preserve dynamic symbols and relocations needed by the loader.
                subprocess.run([strip_command, "--strip-unneeded", str(path)], check=True)
                stripped += 1
    return {"removed_directories": removed, "stripped_libraries": stripped}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_packages", type=Path)
    parser.add_argument("--strip-command")
    args = parser.parse_args()
    print(json.dumps(slim_runtime(args.site_packages, args.strip_command)))
