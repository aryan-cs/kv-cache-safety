# Data

`data/processed/` stores normalized prompt-suite JSONL files produced by `scripts/prepare_data.py`.

`data/external/` is reserved for downloaded public datasets and is ignored by git. Do not commit raw third-party datasets unless their license explicitly permits redistribution and the paper requires a fixed small artifact.

The built-in suites are diagnostic seeds for development and smoke testing. Publication claims should use documented open datasets or clearly label any hand-authored diagnostic suite as non-benchmark evidence.
