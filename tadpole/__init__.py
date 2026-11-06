"""Tadpole — FROG trace preparation and pulse retrieval.

The GUI is Python (PySide6 + matplotlib, styled like Lillypad); the retrieval
algorithms are Julia (julia/TadpoleRetrieval), run in a separate worker
process through juliacall. See ARCHITECTURE.md.
"""

__version__ = "0.1.0"
