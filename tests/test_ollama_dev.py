import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_exploratory_ollama import parser_diagnostic
from src.data.pilot import scenarios
from src.models.ollama_dev import OllamaDevelopmentClient, api, render


class DevelopmentTransportTests(unittest.TestCase):
    def test_network_endpoint_allowlist_prevents_downloads(self):
        for endpoint in ("/api/pull", "/api/push", "https://example.org", "/api/delete"):
            with self.subTest(endpoint=endpoint), patch("urllib.request.urlopen") as request:
                with self.assertRaises(ValueError):
                    api(endpoint)
                request.assert_not_called()

    def test_template_render_is_exact_and_has_no_labels(self):
        prompt = render([{"role": "user", "content": "fixture"}])
        self.assertEqual(prompt, "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nfixture"
                         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_syntax_grammar_and_legality_are_separate(self):
        scenario = scenarios()[0]
        self.assertFalse(parser_diagnostic(scenario, "not JSON")["json_valid"])
        result = parser_diagnostic(scenario, '{"tool":"shell","arguments":{}}')
        self.assertTrue(result["grammar_valid"])
        self.assertFalse(result["legal_action"])
        result = parser_diagnostic(scenario, '{"tool":"check_state","arguments":{}}')
        self.assertTrue(result["legal_action"])

    def test_failed_preflight_never_unloads_preexisting_model(self):
        client = OllamaDevelopmentClient.__new__(OllamaDevelopmentClient)
        client.model, client.owns_residency = "fixture", False
        with patch("src.models.ollama_dev.api", return_value={"models": [{"name": "fixture"}]}) as call:
            self.assertEqual(client.unload(), [{"name": "fixture"}])
            self.assertTrue(all(c.args[0] == "/api/ps" for c in call.call_args_list))

    def test_guard_preserves_reading_on_threshold_failure(self):
        client = OllamaDevelopmentClient.__new__(OllamaDevelopmentClient)
        client.deadline = float("inf")
        client.baseline_memory = {"swap_used_bytes": 0}
        with tempfile.TemporaryDirectory() as directory:
            client.output_dir = Path(directory)
            with patch("src.models.ollama_dev.system_memory", return_value={
                    "memory_free_percentage": 20, "swap_used_bytes": 600 * 1024**2}):
                with self.assertRaisesRegex(RuntimeError, "Swap growth"):
                    client.guard()
            self.assertIn('629145600', (client.output_dir / "memory_guards.jsonl").read_text())
