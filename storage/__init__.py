"""Storage helpers (filesystem JSON, future blob backends)."""

from storage.io import assert_path_exists, read_json_dict

__all__ = ["read_json_dict", "assert_path_exists"]
