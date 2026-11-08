"""Refractive indices from refractiveindex.info, through the `refractiveindex`
package (its database ships inside the package: no network access).

The only module that imports `refractiveindex`, and only on first use — the
import costs ~0.7 s (scipy.interpolate), which startup should not pay.

A page's formula happily evaluates outside the wavelengths it was fitted to
(fused silica's Sellmeier gives n = 1.73 at 100 µm), so every index is NaN
outside the page's own range; physics.py fills those parts and reports them.
"""

from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

_lock = threading.RLock()    # catalogue + material loading, from any thread


@dataclass
class Element:
    """One piece of material in the beam."""
    shelf: str = "main"
    book: str = "SiO2"
    page: str = "Malitson"
    thickness_mm: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Element":
        return cls(str(d["shelf"]), str(d["book"]), str(d["page"]),
                   float(d["thickness_mm"]))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.shelf, self.book, self.page


@dataclass(frozen=True)
class CatalogEntry:
    shelf: str
    book: str
    page: str
    book_name: str           # e.g. "SiO2 (Silicon dioxide, Silica, Quartz)"
    page_name: str           # e.g. "Malitson 1965: n 0.21–6.7 µm"

    @property
    def key(self) -> tuple[str, str, str]:
        return self.shelf, self.book, self.page


def _plain(text: str) -> str:
    return re.sub(r"<[^>]+>", "", str(text or "")).strip()


@lru_cache(maxsize=1)
def catalog() -> tuple[CatalogEntry, ...]:
    """Every page of the database, in the database's own order."""
    with _lock:
        import refractiveindex as ri
        return tuple(CatalogEntry(m.shelf, m.book, m.page, _plain(m.book_name),
                                  _plain(m.name)) for m in ri.search())


def books() -> list[tuple[str, str, str]]:
    """(shelf, book, label) once per book, e.g. ("main", "SiO2",
    "SiO2 (Silicon dioxide, …) — main")."""
    seen, out = set(), []
    for e in catalog():
        if (e.shelf, e.book) not in seen:
            seen.add((e.shelf, e.book))
            name = e.book_name or e.book
            out.append((e.shelf, e.book, name if e.shelf == "main" else f"{name} · {e.shelf}"))
    return out


def pages(shelf: str, book: str) -> list[CatalogEntry]:
    return [e for e in catalog() if e.shelf == shelf and e.book == book]


def entry(shelf: str, book: str, page: str) -> CatalogEntry | None:
    return next((e for e in catalog() if e.key == (shelf, book, page)), None)


@lru_cache(maxsize=256)
def _material(shelf: str, book: str, page: str):
    with _lock:
        import refractiveindex as ri
        try:
            mat = ri.RefractiveIndexMaterial(shelf, book, page)
            lo, hi = mat.get_wl_range(unit="nm", quantity="n")
        except Exception as exc:   # the package raises bare Exception/KeyError
            raise ValueError(f"{book} / {page}: no refractive-index data "
                             f"({exc})") from exc
    return mat, (float(lo), float(hi))


def wl_range(el: Element) -> tuple[float, float]:
    """The wavelengths (nm) the page has data for."""
    return _material(*el.key)[1]


def index_function(el: Element):
    """n(λ [nm]) for this element's page, NaN outside its data range and for
    NaN input."""
    mat, (lo, hi) = _material(*el.key)

    def n_of(wl_nm):
        wl = np.asarray(wl_nm, float)
        ok = np.isfinite(wl) & (wl >= lo) & (wl <= hi)
        out = np.full(wl.shape, np.nan)
        if ok.any():
            with _lock:
                out[ok] = np.real(mat.get_refractive_index(wl[ok], unit="nm"))
        return out
    return n_of


def preload() -> None:
    """Import the package and read the catalogue (for a background thread)."""
    catalog()
