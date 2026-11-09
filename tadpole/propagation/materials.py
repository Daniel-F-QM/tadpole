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


# Which shelves a search shows first: the everyday ones before the exotic.
SHELF_ORDER = {"main": 0, "popular_glass": 1, "specs": 2, "glass": 3,
               "organic": 4, "other": 5, "3d": 6}
# Shelves whose "book" is a catalogue and whose pages are the glasses
# (specs/SCHOTT-optical/N-BK7): the page is the name people type.
_CATALOGUE_SHELVES = {"specs"}


def search_label(e: CatalogEntry) -> str:
    """One line per data set, name first: "SiO2 — Malitson 1965: n 0.21–6.7 µm",
    "N-BK7 — SCHOTT - Optical glass · specs"."""
    if e.shelf in _CATALOGUE_SHELVES:
        text = f"{e.page} — {e.book_name or e.book}"
    else:
        text = f"{e.book} — {e.page_name or e.page}"
    return text if e.shelf == "main" else f"{text} · {e.shelf}"


def _k_only(page_name: str) -> bool:
    """Data sets that hold only the extinction coefficient k ("… 2017: Water;
    k 1.2–1.9 µm") — no n, so no dispersion. Judged from the name, which
    the database writes as "<source>: <what>; <quantity> <range>"."""
    if ":" not in page_name:
        return False
    starts = [p.strip().split(" ")[0] for p in page_name.split(":", 1)[1].split(";")]
    return "k" in starts and not any(s.startswith("n") for s in starts)


def search(query: str, limit: int = 40) -> list[CatalogEntry]:
    """Data sets with a refractive index matching every word of `query`
    (case-insensitive, anywhere in the material's id, name, data set or
    shelf). Order: exact names, then common shelves before exotic ones, then
    names starting with the first word, then the database's order."""
    words = query.lower().split()
    if not words:
        return []
    first = words[0]
    hits = []
    for i, e in enumerate(catalog()):
        hay = f"{e.book} {e.page} {e.book_name} {e.page_name} {e.shelf}".lower()
        if not all(w in hay for w in words) or _k_only(e.page_name):
            continue
        names = (e.book.lower(), e.page.lower())
        exact = 0 if first in names else 1
        prefix = 0 if any(n.startswith(first) for n in names) else 1
        # the "3d" shelf (pages named after the material, for 3-D rendering)
        # would otherwise win every exact match against the real entries
        last = 1 if e.shelf == "3d" else 0
        hits.append((last, exact, SHELF_ORDER.get(e.shelf, 9), prefix, i, e))
    hits.sort(key=lambda h: h[:5])
    return [h[5] for h in hits[:limit]]


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
