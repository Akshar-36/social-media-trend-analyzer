"""Packaging regressions using disposable files only; no ML dependencies needed."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "slim_runtime", Path(__file__).resolve().parents[1] / "container/slim_runtime.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SlimRuntimeTests(unittest.TestCase):
    def test_preserves_runtime_code_metadata_models_and_testing_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keep = ["torch/__init__.py", "torch/testing/__init__.py", "weights/model.safetensors",
                    "model.joblib", "package.dist-info/METADATA", "lib/libreal.so"]
            remove = ["pkg/tests/nested/test_case.py", "pkg/test/fixture.bin",
                      "pkg/__pycache__/module.pyc", "torch/include/header.h",
                      "numpy/_core/include/header.h", "pyarrow/include/header.h"]
            for name in keep + remove:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"unchanged")
            module.slim_runtime(root)
            for name in keep:
                self.assertEqual((root / name).read_bytes(), b"unchanged", name)
            for name in remove:
                self.assertFalse((root / name).exists(), name)

    def test_strips_only_elf_libraries_and_fails_on_strip_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "real.so.1"
            library.write_bytes(b"\x7fELFbinary")
            (root / "not-elf.so").write_bytes(b"text")
            (root / "weights.bin").write_bytes(b"\x7fELFmodel")
            with patch.object(module.subprocess, "run") as run:
                module.slim_runtime(root, "strip")
                run.assert_called_once_with(["strip", "--strip-unneeded", str(library)], check=True)
            with patch.object(module.subprocess, "run", side_effect=RuntimeError("failed")):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    module.slim_runtime(root, "strip")


if __name__ == "__main__":
    unittest.main()
