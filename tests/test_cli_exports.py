import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llm-hardware-estimator"))

from estimator.config_loader import load_mapping_file, validate_project_configs
from estimator.export_xlsx import write_xlsx


class ConfigAndExportTests(unittest.TestCase):
    def test_default_project_configs_validate(self):
        report = validate_project_configs(ROOT / "llm-hardware-estimator" / "configs")

        self.assertEqual(report["status"], "ok")
        self.assertGreaterEqual(report["models"], 3)
        self.assertGreaterEqual(report["hardware"], 3)

    def test_json_compatible_yaml_loader_reads_nested_config(self):
        models = load_mapping_file(ROOT / "llm-hardware-estimator" / "configs" / "models.yaml")

        self.assertIn("models", models)
        self.assertIn("qwen3-32b", models["models"])

    def test_xlsx_writer_creates_valid_zip_workbook(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report.xlsx"
            write_xlsx(
                target,
                {
                    "Summary": [{"metric": "status", "value": "ok"}],
                    "GPU_Comparison": [{"rank": 1, "gpu_model": "96G"}],
                },
            )

            self.assertGreater(target.stat().st_size, 1000)
            with target.open("rb") as fh:
                self.assertEqual(fh.read(2), b"PK")


class CliSmokeTests(unittest.TestCase):
    def test_recommend_command_exports_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "report.md"
            json_path = Path(tmp) / "result.json"
            proc = subprocess.run(
                [
                    "python3",
                    str(ROOT / "llm-hardware-estimator" / "estimator.py"),
                    "recommend",
                    "--model",
                    "qwen3-32b",
                    "--scenario",
                    "rag",
                    "--concurrency",
                    "4",
                    "--export-md",
                    str(md_path),
                    "--export-json",
                    str(json_path),
                ],
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("推荐", proc.stdout)
            self.assertTrue(md_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_id"], "qwen3-32b")
            self.assertIn("gpu_candidates", payload)

    def test_invalid_model_returns_similar_choices(self):
        proc = subprocess.run(
            [
                "python3",
                str(ROOT / "llm-hardware-estimator" / "estimator.py"),
                "estimate",
                "--model",
                "qwen32",
                "--hardware",
                "mthreads-s4000",
                "--scenario",
                "rag",
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("qwen3-32b", proc.stderr)


if __name__ == "__main__":
    unittest.main()
