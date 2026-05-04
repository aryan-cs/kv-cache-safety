import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from check_prepared_suites import check_prepared_suites, parse_suite_min_records

from cache_safety_erasure.utils.io import file_sha256, write_json


def test_check_prepared_suites_validates_counts_hashes_and_public_provenance(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    suite_path = data_dir / "public_refusal_safety.jsonl"
    suite_path.write_text(
        json.dumps(
            {
                "id": "p1",
                "suite": "public_refusal_safety",
                "metadata": {
                    "source_dataset": "dataset",
                    "source_config": None,
                    "source_config_name": "default",
                    "source_revision": "abc123",
                    "source_fingerprint": "fingerprint",
                    "source_split": "train",
                    "source_version": "1.0.0",
                    "source_homepage": "",
                    "source_license": "",
                    "source_row_index": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        data_dir / "public_refusal_safety.manifest.json",
        {
            "suite_name": "public_refusal_safety",
            "record_count": 1,
            "sha256": file_sha256(suite_path),
        },
    )

    failures = check_prepared_suites(
        suites=["public_refusal_safety"],
        data_dir=data_dir,
        min_records=10,
        suite_min_records={"public_refusal_safety": 1},
        require_public_provenance=True,
    )

    assert failures == []


def test_check_prepared_suites_reports_publication_blockers(tmp_path: Path) -> None:
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    suite_path = data_dir / "public_refusal_safety.jsonl"
    suite_path.write_text(
        '{"id":"p1","suite":"public_refusal_safety","metadata":{}}\n',
        encoding="utf-8",
    )
    write_json(
        data_dir / "public_refusal_safety.manifest.json",
        {"suite_name": "wrong", "record_count": 2, "sha256": "stale"},
    )

    failures = check_prepared_suites(
        suites=["public_refusal_safety"],
        data_dir=data_dir,
        min_records=2,
        suite_min_records={},
        require_public_provenance=True,
    )

    assert any("has 1 records; need >= 2" in failure for failure in failures)
    assert any("manifest suite_name" in failure for failure in failures)
    assert any("record_count" in failure for failure in failures)
    assert any("sha256 is stale" in failure for failure in failures)
    assert any("without dataset provenance" in failure for failure in failures)


def test_parse_suite_min_records_requires_suite_equals_count() -> None:
    assert parse_suite_min_records(["system_leakage=2"]) == {"system_leakage": 2}
