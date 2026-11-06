"""What the window currently holds. Plain attributes, no Qt."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppState:
    raw: object = None            # tadpole.io.dwc.RawTrace
    processed: tuple | None = None  # (trace, wl, delays) after crop + background
    prepared: object = None       # tadpole.prep.model.PreparedTrace (on Prepare page)
    report: object = None         # tadpole.prep.pipeline.PrepReport
    recon_input: object = None    # PreparedTrace sent to the Reconstruct page
    result: dict | None = None    # last finished retrieval (engine Done payload)
