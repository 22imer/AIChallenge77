"""Run with: python -m unittest discover -s tests -v."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

NOTEBOOK = Path(__file__).resolve().parents[1] / "finetune_colab.ipynb"


class ParallelDataTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(NOTEBOOK.exists(), "Standalone fine-tuning notebook is missing")
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cell = next(c for c in notebook["cells"] if c["id"] == "data-functions")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        module_path = self.root / "notebook_data.py"
        module_path.write_text("".join(cell["source"]), encoding="utf-8")
        spec = importlib.util.spec_from_file_location("notebook_data", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.ns = vars(module)

    def parallel(self, src, tgt):
        a, b = self.root / "train.zh", self.root / "train.vi"
        a.write_text(src, encoding="utf-8")
        b.write_text(tgt, encoding="utf-8")
        return self.ns["read_parallel"](a, b)

    def test_blank_pair_does_not_shift_later_translation(self):
        pairs = self.parallel("甲\n\n乙\n甲\n", "mot\nbo qua\nhai\nmot\n")
        self.assertEqual(pairs, [("甲", "mot"), ("乙", "hai")])

    def test_mismatched_line_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            self.parallel("甲\n乙\n", "mot\n")

    def test_all_translations_of_one_source_stay_in_one_partition(self):
        pairs = [("甲", "mot"), ("甲", "thu nhat"), ("乙", "hai"),
                 ("丙", "ba"), ("丁", "bon"), ("戊", "nam")]
        train, valid = self.ns["split_pairs"](pairs, valid_fraction=0.4, seed=42)
        self.assertFalse({s for s, _ in train} & {s for s, _ in valid})
        self.assertEqual(sorted(train + valid), sorted(pairs))
        self.assertTrue(train)
        self.assertTrue(valid)
        self.assertEqual((train, valid), self.ns["split_pairs"](pairs, 0.4, 42))

    def test_single_source_cannot_create_leak_free_validation(self):
        with self.assertRaises(ValueError):
            self.ns["split_pairs"]([("甲", "mot"), ("甲", "thu nhat")])


class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cell = next(c for c in notebook["cells"] if c["id"] == "checkpoint-functions")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        module_path = self.root / "notebook_checkpoint.py"
        module_path.write_text(
            "import json, os\nfrom pathlib import Path\n" + "".join(cell["source"]),
            encoding="utf-8",
        )
        spec = importlib.util.spec_from_file_location("notebook_checkpoint", module_path)
        assert spec is not None and spec.loader is not None
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def snapshot(self, name, complete=True):
        path = self.root / name
        path.mkdir()
        if complete:
            (path / "complete.json").write_text("{}", encoding="utf-8")
        return path

    def test_interrupted_publication_preserves_both_previous_selections(self):
        old = self.snapshot("checkpoint-old")
        new = self.snapshot("checkpoint-new")
        self.module.publish_checkpoint(self.root, old.name, is_best=True)
        # Simulate interruption at the filesystem commit boundary, not model behavior.
        with (
            patch.object(self.module.os, "replace", side_effect=OSError("interrupted")),
            self.assertRaises(OSError),
        ):
            self.module.publish_checkpoint(self.root, new.name, is_best=True)
        self.assertEqual(self.module.checkpoint_from_pointer(self.root), old)
        self.assertEqual(self.module.checkpoint_from_pointer(self.root, "best"), old)

    def test_nonwinning_snapshot_keeps_best_and_winner_updates_both(self):
        old = self.snapshot("checkpoint-old")
        latest = self.snapshot("checkpoint-latest")
        winner = self.snapshot("checkpoint-winner")
        self.module.publish_checkpoint(self.root, old.name, is_best=True)
        self.module.publish_checkpoint(self.root, latest.name)
        self.assertEqual(self.module.checkpoint_from_pointer(self.root), latest)
        self.assertEqual(self.module.checkpoint_from_pointer(self.root, "best"), old)
        self.module.publish_checkpoint(self.root, winner.name, is_best=True)
        self.assertEqual(self.module.checkpoint_from_pointer(self.root), winner)
        self.assertEqual(self.module.checkpoint_from_pointer(self.root, "best"), winner)

    def test_incomplete_latest_falls_back_without_silently_restarting(self):
        old = self.snapshot("checkpoint-old")
        incomplete = self.snapshot("checkpoint-incomplete", complete=False)
        self.module.write_json_atomic(
            self.root / "checkpoint-index.json",
            {"latest": [incomplete.name, old.name], "best": [old.name]},
        )
        self.assertEqual(self.module.checkpoint_from_pointer(self.root), old)
        (old / "complete.json").unlink()
        with self.assertRaises(RuntimeError):
            self.module.checkpoint_from_pointer(self.root)


if __name__ == "__main__":
    unittest.main()
