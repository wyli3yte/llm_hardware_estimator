import importlib.util
import json
import sys
import tempfile
import unittest
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llm-hardware-estimator"))


def load_web_app():
    path = ROOT / "llm-hardware-estimator" / "web_app.py"
    spec = importlib.util.spec_from_file_location("web_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WebPortalTests(unittest.TestCase):
    def test_build_recommendation_response_writes_downloadable_xlsx(self):
        web_app = load_web_app()

        with tempfile.TemporaryDirectory() as tmp:
            result = web_app.build_recommendation_response(
                {
                    "model": "qwen3-32b",
                    "scenario": "rag",
                    "concurrency": 4,
                    "goal": "balanced",
                    "target": "domestic",
                },
                output_dir=Path(tmp),
            )

            self.assertEqual(result["model_id"], "qwen3-32b")
            self.assertIn("decision", result)
            self.assertTrue(result["gpu_candidates"])
            self.assertTrue(result["download_url"].startswith("/download/"))
            workbook_path = Path(tmp) / result["download_url"].split("/")[-1]
            self.assertTrue(workbook_path.exists())
            with zipfile.ZipFile(workbook_path) as workbook:
                workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")
            self.assertIn('name="Decision"', workbook_xml)

    def test_precision_selection_applies_quantization_bundle(self):
        web_app = load_web_app()

        with tempfile.TemporaryDirectory() as tmp:
            bf16 = web_app.build_recommendation_response(
                {"model": "qwen3-32b", "scenario": "rag", "weight_precision": "bf16"},
                output_dir=Path(tmp),
            )
            w4a16 = web_app.build_recommendation_response(
                {"model": "qwen3-32b", "scenario": "rag", "weight_precision": "w4a16"},
                output_dir=Path(tmp),
            )

            bf16_memory = float(bf16["decision"]["memory"].replace(" GB", ""))
            w4a16_memory = float(w4a16["decision"]["memory"].replace(" GB", ""))
            self.assertLess(w4a16_memory, bf16_memory)
            self.assertAlmostEqual(w4a16_memory, 33.36, places=1)
            workbook_path = Path(tmp) / w4a16["download_url"].split("/")[-1]
            with zipfile.ZipFile(workbook_path) as workbook:
                decision_xml = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")
            self.assertIn("w4a16 / KV bf16", decision_xml)

    def test_options_include_kv_cache_quantization_choice(self):
        web_app = load_web_app()

        options = web_app.options_response()

        self.assertIn("w4a16_kv_int8", options["precisions"])

    def test_homepage_is_hardware_estimator_focused(self):
        homepage = (ROOT / "llm-hardware-estimator" / "web" / "index.html").read_text(encoding="utf-8")

        self.assertIn("LLM推理硬件需求估算与国产GPU选型工具", homepage)
        self.assertIn("生成方案", homepage)

    def test_http_server_serves_options_and_recommendation(self):
        web_app = load_web_app()

        with tempfile.TemporaryDirectory() as tmp:
            server = web_app.create_server("127.0.0.1", 0, output_dir=Path(tmp))
            port = server.server_address[1]
            try:
                web_app.start_server_thread(server)

                with urllib.request.urlopen("http://127.0.0.1:%d/api/options" % port, timeout=5) as resp:
                    options = json.loads(resp.read().decode("utf-8"))
                self.assertIn("qwen3-32b", options["models"])
                self.assertIn("rag", options["scenarios"])

                req = urllib.request.Request(
                    "http://127.0.0.1:%d/api/recommend" % port,
                    data=json.dumps({"model": "qwen3-32b", "scenario": "rag"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(result["download_url"])
                self.assertTrue(result["gpu_candidates"])
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
