from __future__ import annotations

from pathlib import Path

from importlib import resources

test_data_dir = Path(resources.files("gnn_tracking.test_data"))  # type: ignore

trackml_test_data_dir = test_data_dir / "trackml"
trackml_test_data_prefix = "test_event000000001"
