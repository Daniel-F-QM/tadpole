"""Plot styling: the matplotlib backend, the global rcParams, and the figure
geometry the canvas lays itself out with.

Importing this module SELECTS THE BACKEND. That is an import-time side effect, and
a deliberate one: every module that builds any part of the figure imports this
first, above `from matplotlib.figure import Figure`, so the backend is chosen
before matplotlib can pick one for itself. Module import is idempotent, so there
is no "the bootstrap must run first" rule to remember.

The colours the canvas draws with are NOT here — they come from
design.tokens.PALETTE, read at paint time so a theme switch reaches artists that
already exist.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")
matplotlib.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9.0, "axes.titleweight": "bold", "axes.titlesize": 9.5,
    "axes.labelsize": 9,
})

# Colormaps offered for the FROG trace; the FIRST ENTRY IS THE DEFAULT, so the
# order is load-bearing.
#
# "jet" leads because it is what the group uses most, NOT because it is the best
# choice — it is the one entry here that is not perceptually uniform. The other
# five map equal steps in count to equal steps in apparent brightness; jet does
# not, so a trace read off it shows structure the data does not have. It is the
# default because matching what colleagues are used to reading matters more here
# than the colour science, and that is a deliberate trade, not an oversight.
TRACE_COLORMAPS = ["jet", "magma", "viridis", "inferno", "plasma", "cividis"]


# Colour bar beside a trace image, in pixels (it holds fixed-size text).
CBAR_W_PX = 12

# Resolution the toolbar's camera prints at (the figure is printed at its
# on-screen size; this is a detail multiplier), capped so a 4K window does not
# produce a gigantic PNG.
PICTURE_DPI = 300
PICTURE_MAX_PX = 6000
