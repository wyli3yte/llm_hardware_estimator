import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llm-hardware-estimator"))

from estimator.memory_model import estimate_memory
from estimator.precision import bytes_per_param, parse_precision_bundle
from estimator.hardware_selector import recommend_hardware


QWEN32B = {
    "model_type": "dense",
    "task_type": "generation",
    "params_billion": 32,
    "active_params_billion": 32,
    "num_layers": 64,
    "hidden_size": 5120,
    "num_attention_heads": 40,
    "num_kv_heads": 8,
    "attention_type": "full",
    "sliding_window_size": None,
    "max_context_length": 131072,
    "confidence": "medium",
}

RAG_SCENARIO = {
    "input_tokens": 4096,
    "output_tokens": 512,
    "concurrency": 4,
    "weight_precision": "bf16",
    "activation_precision": "bf16",
    "kv_cache_precision": "bf16",
    "quantization": "none",
    "kv_cache_mode": "peak",
    "gpu_utilization_target": 0.85,
    "runtime_overhead": {
        "absolute_gb": 2.0,
        "weight_ratio": 0.10,
        "kv_cache_ratio": 0.15,
    },
    "production_margin_ratio": 0.30,
    "parallel": {
        "strategy": "none",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "parallel_overhead_factor": 1.00,
    },
    "decision_goal": "balanced",
}


class PrecisionTests(unittest.TestCase):
    def test_precision_bundle_handles_weight_and_kv_quantization(self):
        bundle = parse_precision_bundle("w4a16_kv_int8")

        self.assertEqual(bundle.weight_precision, "w4a16")
        self.assertEqual(bundle.kv_cache_precision, "int8")
        self.assertEqual(bytes_per_param(bundle.weight_precision), 0.5)

    def test_weight_only_quantization_keeps_kv_cache_bf16(self):
        bundle = parse_precision_bundle("w4a16")

        self.assertEqual(bundle.weight_precision, "w4a16")
        self.assertEqual(bundle.activation_precision, "bf16")
        self.assertEqual(bundle.kv_cache_precision, "bf16")


class MemoryEstimatorTests(unittest.TestCase):
    def test_generation_memory_matches_design_formula(self):
        result = estimate_memory(QWEN32B, RAG_SCENARIO)

        self.assertTrue(result.supported_parallel_config)
        self.assertAlmostEqual(result.weight_memory_gb, 59.60, places=2)
        self.assertAlmostEqual(result.kv_cache_peak_gb, 4.50, places=2)
        self.assertAlmostEqual(result.kv_cache_avg_gb, 4.25, places=2)
        self.assertAlmostEqual(result.production_memory_gb, 91.96, places=2)
        self.assertEqual(result.task_type, "generation")

    def test_tensor_parallel_rejects_non_divisible_kv_heads(self):
        scenario = dict(RAG_SCENARIO)
        scenario["parallel"] = {
            "strategy": "tensor_parallel",
            "tensor_parallel_size": 3,
            "pipeline_parallel_size": 1,
            "parallel_overhead_factor": 1.10,
        }

        result = estimate_memory(QWEN32B, scenario)

        self.assertFalse(result.supported_parallel_config)
        self.assertIn("num_kv_heads", result.warnings[0])

    def test_embedding_does_not_allocate_kv_cache(self):
        model = {
            "model_type": "encoder_or_embedding",
            "task_type": "embedding",
            "params_billion": 8,
            "active_params_billion": 8,
            "max_context_length": 40960,
            "confidence": "low",
        }
        scenario = dict(RAG_SCENARIO)
        scenario["output_tokens"] = 0

        result = estimate_memory(model, scenario)

        self.assertEqual(result.kv_cache_peak_gb, 0)
        self.assertGreater(result.production_memory_gb, result.weight_memory_gb)
        self.assertEqual(result.task_type, "embedding")


class HardwareSelectorTests(unittest.TestCase):
    def test_recommendation_prefers_fit_with_better_margin_and_confidence(self):
        hardware = {
            "small-48g": {
                "vendor": "测试厂商",
                "model": "48G",
                "region_type": "domestic",
                "memory_gb": 48,
                "memory_bandwidth_gbps": 768,
                "fp16_tflops": 100,
                "bf16_tflops": 100,
                "int8_tops": 200,
                "precision_support": ["fp16", "bf16", "int8"],
                "software_profile": "test",
                "source_type": "official",
                "confidence": "high",
            },
            "large-96g": {
                "vendor": "测试厂商",
                "model": "96G",
                "region_type": "domestic",
                "memory_gb": 96,
                "memory_bandwidth_gbps": 1600,
                "fp16_tflops": 220,
                "bf16_tflops": 220,
                "int8_tops": 440,
                "precision_support": ["fp16", "bf16", "int8"],
                "software_profile": "test",
                "source_type": "official",
                "confidence": "high",
            },
        }

        ranked = recommend_hardware(QWEN32B, RAG_SCENARIO, hardware, goal="balanced")

        self.assertEqual(ranked[0].hardware_id, "large-96g")
        self.assertEqual(ranked[0].fit_status, "Fit")
        self.assertTrue(math.isfinite(ranked[0].score))
        self.assertGreaterEqual(ranked[1].suggested_gpu_count, 2)


if __name__ == "__main__":
    unittest.main()
