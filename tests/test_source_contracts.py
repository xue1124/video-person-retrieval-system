"""不启动模型、Redis或MySQL也能运行的源码结构检查。"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SourceContractsTest(unittest.TestCase):
    def test_all_python_files_parse(self) -> None:
        files = [
            path
            for path in ROOT.rglob("*.py")
            if not {".venv", "venv", "node_modules"}.intersection(path.parts)
        ]
        self.assertGreater(len(files), 0)
        for path in files:
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    def test_service_layer_does_not_import_root_tasks(self) -> None:
        offenders: list[str] = []
        for path in (ROOT / "services").rglob("*.py"):
            source = path.read_text(encoding="utf-8-sig")
            if re.search(r"^\s*(?:from\s+tasks\s+import|import\s+tasks\b)", source, re.M):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual([], offenders)

    def test_schema_contains_expected_tables(self) -> None:
        source = (ROOT / "sql" / "schema.sql").read_text(encoding="utf-8-sig")
        tables = set(
            re.findall(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+`?([a-z_]+)`?", source, re.I)
        )
        expected = {
            "videos",
            "processing_runs",
            "global_people",
            "video_people",
            "tracks",
            "track_points",
            "observations",
            "observation_embeddings",
            "gallery_meta",
            "users",
            "search_logs",
            "analysis_reports",
        }
        self.assertTrue(expected.issubset(tables), expected - tables)
        self.assertEqual(17, len(tables))

    def test_example_environment_has_core_settings(self) -> None:
        source = (ROOT / "deploy" / "env" / "siglip.env.example").read_text(
            encoding="utf-8-sig"
        )
        for name in (
            "TRACKING_DB_URL",
            "JWT_SECRET",
            "CELERY_BROKER_URL",
            "CELERY_RESULT_BACKEND",
            "CORS_ORIGINS",
        ):
            with self.subTest(name=name):
                self.assertRegex(source, rf"(?m)^{name}=")


if __name__ == "__main__":
    unittest.main()
