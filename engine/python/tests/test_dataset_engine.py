import json
import tempfile
import unittest
from pathlib import Path

from bella_dataset.cli import normalize_record, prepare


class NormalizeTests(unittest.TestCase):
    def test_instruction_input_output(self):
        result = normalize_record({"instruction": "Add two numbers", "input": "2 + 3", "output": "5"})
        self.assertEqual(result[0]["role"], "user")
        self.assertIn("2 + 3", result[0]["content"])
        self.assertEqual(result[1]["content"], "5")

    def test_sharegpt_conversations(self):
        result = normalize_record({"conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi"}
        ]})
        self.assertEqual([m["role"] for m in result], ["user", "assistant"])

    def test_unrecognized_record_is_skipped(self):
        self.assertIsNone(normalize_record({"code": "print('hello')"}))


class PipelineTests(unittest.TestCase):
    def test_heldout_duplicate_is_not_written_to_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            output = Path(tmp) / "output"
            (root / "training").mkdir(parents=True)
            (root / "test").mkdir()
            record = {"prompt": "Shared example", "completion": "Shared answer"}
            (root / "training" / "train.jsonl").write_text(json.dumps(record) + "\\n", encoding="utf-8")
            (root / "test" / "test.jsonl").write_text(json.dumps(record) + "\\n", encoding="utf-8")

            manifest = prepare(root, output)

            self.assertEqual(manifest["counts"]["train"], 0)
            self.assertEqual(manifest["counts"]["heldout"], 1)
            self.assertEqual(manifest["counts"]["duplicates"], 1)
            heldout = json.loads((output / "heldout.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(heldout["source_file"], "test/test.jsonl")

    def test_malformed_jsonl_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            output = Path(tmp) / "output"
            root.mkdir()
            (root / "records.jsonl").write_text(
                '{"prompt":"valid","completion":"record"}\\n{not-json}\\n',
                encoding="utf-8"
            )

            manifest = prepare(root, output)

            self.assertEqual(manifest["counts"]["malformed_jsonl_lines"], 1)
            self.assertEqual(manifest["counts"]["normalized"], 1)

    def test_train_heldout_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            output = Path(tmp) / "output"
            (root / "training").mkdir(parents=True)
            (root / "test").mkdir()
            record = {"prompt": "What is 1+1?", "completion": "2"}
            (root / "training" / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            (root / "test" / "test.jsonl").write_text(json.dumps({"prompt": "What is 2+2?", "completion": "4"}) + "\n", encoding="utf-8")
            manifest = prepare(root, output)
            self.assertEqual(manifest["counts"]["train"], 1)
            self.assertEqual(manifest["counts"]["heldout"], 1)
            self.assertTrue((output / "manifest.json").exists())
            train = (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
            heldout = (output / "heldout.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(train), 1)
            self.assertEqual(len(heldout), 1)


if __name__ == "__main__":
    unittest.main()
