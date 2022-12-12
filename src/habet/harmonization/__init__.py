from pathlib import Path

to_ignore = ["__init__"]
g = Path(__file__).parents[0].glob("*.py")
__all__ = [p.stem for p in g if p.is_file() and p.stem not in to_ignore]