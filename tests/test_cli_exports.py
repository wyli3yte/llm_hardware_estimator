import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
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
        self.assertGreaterEqual(report["scenarios"], 6)

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
    def test_output_xlsx_creates_focused_single_workbook(self):
        with tempfile.TemporaryDirectory() as tmp:
            xlsx_path = Path(tmp) / "decision.xlsx"
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
                    "--output",
                    str(xlsx_path),
                ],
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(xlsx_path.exists())
            with zipfile.ZipFile(xlsx_path) as workbook:
                workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")
                sheet1_xml = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")

            for sheet_name in ("Decision", "Options", "GPU_Comparison", "What_If", "Appendix"):
                self.assertIn('name="%s"' % sheet_name, workbook_xml)
            self.assertNotIn('name="Summary"', workbook_xml)
            self.assertIn("LLM推理硬件需求估算与国产GPU选型报告", sheet1_xml)
            self.assertIn("面向模型容量估算、国产GPU初筛和测试验证", sheet1_xml)
            self.assertIn("推荐结论", sheet1_xml)
            self.assertIn("关键风险Top3", sheet1_xml)
            with zipfile.ZipFile(xlsx_path) as workbook:
                appendix_xml = workbook.read("xl/worksheets/sheet5.xml").decode("utf-8")
            self.assertIn("显存误差≤5%：通过", appendix_xml)
            self.assertIn("5%-10%：可接受", appendix_xml)
            self.assertIn("&gt;10%：不通过", appendix_xml)

    def test_recommend_output_xlsx_includes_precision_and_scenario_what_if(self):
        with tempfile.TemporaryDirectory() as tmp:
            xlsx_path = Path(tmp) / "decision.xlsx"
            proc = subprocess.run(
                [
                    "python3",
                    str(ROOT / "llm-hardware-estimator" / "estimator.py"),
                    "recommend",
                    "--model",
                    "qwen3-32b",
                    "--scenario",
                    "rag",
                    "--output",
                    str(xlsx_path),
                ],
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            with zipfile.ZipFile(xlsx_path) as workbook:
                what_if_xml = workbook.read("xl/worksheets/sheet4.xml").decode("utf-8")

            self.assertIn("precision", what_if_xml)
            self.assertIn("scenario", what_if_xml)
            self.assertIn("w4a16_kv_int8", what_if_xml)

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
            self.assertIn("scenario_comparison", payload)
            self.assertIn("sources", payload)
            self.assertTrue(payload["scenario_comparison"])
            self.assertTrue(payload["sources"])

    def test_recommend_command_exports_scenario_and_source_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_dir = Path(tmp) / "csv"
            proc = subprocess.run(
                [
                    "python3",
                    str(ROOT / "llm-hardware-estimator" / "estimator.py"),
                    "recommend",
                    "--model",
                    "qwen3-32b",
                    "--scenario",
                    "rag",
                    "--export-csv",
                    str(csv_dir),
                ],
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            scenario_csv = csv_dir / "scenario_comparison.csv"
            sources_csv = csv_dir / "sources.csv"
            self.assertTrue(scenario_csv.exists())
            self.assertTrue(sources_csv.exists())
            self.assertIn("long_doc", scenario_csv.read_text(encoding="utf-8"))
            self.assertIn("source_url", sources_csv.read_text(encoding="utf-8"))

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
