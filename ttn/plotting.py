"""Figure style and plotting helpers for report figures (matplotlib; works with the Agg backend).

Conventions
- Call `set_style()` once per script, before creating figures. Sizes assume a LaTeX text width of
  6.5 in (`TEXT_WIDTH`): make each figure at its final size and include it at natural size
  (\\includegraphics{fig.pdf}, no width=), so 8-9 pt figure text sits next to 10-11 pt body text.
  Text and math use STIX (Times-like, bundled with matplotlib, so a figure looks the same on every
  machine). A report in Times-like type matches it closely: \\usepackage{stix2} (the same design)
  or newtxtext/newtxmath; LaTeX's default Computer Modern does not.
- Colour does one job per figure. The palettes are the dataviz skill's reference palette, checked
  with its validator against a white surface (Delta E = OKLab distance x 100; CVD = protanopia and
  deuteranopia simulated with Machado et al. 2009):
  * identity (teacher component r, singular-value index k, tree shape): `palette()`, eight hues
    in a fixed order, assigned in order and never cycled. Adjacent slots keep CVD Delta E >= 9.1
    and normal-vision Delta E >= 19.6. Slots 3-5 (aqua, yellow, magenta) have < 3:1 contrast on
    white, so plots with several series always carry a legend.
  * an ordered sweep (one curve per sigma, chi, K, ...): `ordinal_colors(n)`, one hue light -> dark.
  * magnitude in a heatmap: `sequential_cmap()`, one hue, light = small, dark = large.
  * regimes in a phase diagram: `regime_colors()`. Any two cells can touch, so every pair (not only
    adjacent slots) must stay distinct: up to five regimes keep CVD Delta E >= 13.0 and normal-vision
    Delta E >= 16.3 for all pairs.
- t is gradient-flow time. Data may be numpy arrays, torch tensors or lists of them (e.g. one tensor
  per saved time point); everything is converted to float64 numpy arrays.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_hex, to_rgb
from matplotlib.figure import FigureBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import linregress

TEXT_WIDTH = 6.5  # inches

# blue, orange, aqua, yellow, magenta, green, violet, red. The order is what keeps adjacent slots
# apart under colour-vision deficiency, so never reorder or cycle it.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
# blue, yellow, green, magenta, violet: the subset whose every prefix passes the all-pairs checks.
REGIME_PALETTE = ("#2a78d6", "#eda100", "#008300", "#e87ba4", "#4a3aa7")
# One-hue blue ramp, steps 100, 150, ..., 700.
BLUE_RAMP = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
INK = "#0b0b0b"  # titles and axis labels
INK_SECONDARY = "#52514e"  # tick labels
MUTED = "#898781"  # spines, ticks, reference lines, de-emphasised series
GRID = "#e1e0d9"
SURFACE = "#ffffff"


# ---- colours ---------------------------------------------------------------------------------


def palette(n: int | None = None) -> list[str]:
    """The first n categorical colours (all eight if n is None), in their fixed order.

    There is no ninth colour: fold extra series into "other" or use small multiples, since a
    generated ninth hue cannot stay distinct from the others under colour-vision deficiency.
    """
    if n is not None and n > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} categorical colours, got n = {n}")
    return list(PALETTE[:n])


def ordinal_colors(n: int) -> list[str]:
    """n colours from light to dark along the blue ramp (steps 250 -> 700), one per value of an
    ordered sweep, in sweep order. For n <= 7 they pass the dataviz ordinal checks (neighbours
    differ by >= 0.06 in OKLCH lightness; the light end has 2.1:1 contrast on white). Beyond 7,
    neighbours blur: use a colorbar instead."""
    if n == 1:
        return [BLUE_RAMP[-1]]
    cmap = LinearSegmentedColormap.from_list("ordinal", BLUE_RAMP[3:])
    return [to_hex(cmap(x)) for x in np.linspace(0.0, 1.0, n)]


def sequential_cmap() -> LinearSegmentedColormap:
    """One-hue blue colormap for magnitudes (light = small, dark = large); NaN cells stay blank."""
    return LinearSegmentedColormap.from_list("ttn_blue", BLUE_RAMP).with_extremes(bad="none")


def regime_colors(categories) -> dict:
    """{category: colour} in the given order; pass the same ordered list to every figure so that a
    regime keeps its colour. Up to five categories use `REGIME_PALETTE` (every pair distinct).
    Six to eight fall back to `PALETTE`, whose non-adjacent pairs can collide under colour-vision
    deficiency, so label the cells as well (`plot_phase_diagram(..., annotate=True)`)."""
    categories = list(categories)
    if len(categories) <= len(REGIME_PALETTE):
        return dict(zip(categories, REGIME_PALETTE))
    return dict(zip(categories, palette(len(categories))))


# ---- style -----------------------------------------------------------------------------------


def set_style(font: str = "serif") -> None:
    """Set rcParams for report figures.

    font="serif": STIX text and math (Times-like). font="sans": DejaVu Sans text and math (slides).
    Computer Modern (matplotlib's cmr10) is not offered: it is TeX-encoded, so plain-text {, }, |,
    _ and ^ silently render as other glyphs.
    """
    if "ttn_blue" not in mpl.colormaps:
        mpl.colormaps.register(sequential_cmap(), name="ttn_blue")
    fonts = {
        "serif": {"font.family": "STIXGeneral", "mathtext.fontset": "stix"},
        "sans": {"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"},
    }
    mpl.rcParams.update(fonts[font])
    mpl.rcParams.update(
        {
            # layout and output
            "figure.figsize": (TEXT_WIDTH, 0.4 * TEXT_WIDTH),
            "figure.constrained_layout.use": True,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # TrueType instead of Type 3 fonts: searchable text, accepted by journals and arXiv.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # text
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "figure.titlesize": 9,
            "figure.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlelocation": "left",
            "axes.formatter.use_mathtext": True,
            # quiet chrome: thin muted spines and ticks, hairline grid behind the data
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            # marks
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "image.cmap": "ttn_blue",
            "lines.linewidth": 1.5,
            "lines.markersize": 5.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "errorbar.capsize": 2.0,
            "legend.frameon": False,
            "legend.handlelength": 2.0,
            "legend.labelspacing": 0.3,
            "legend.borderaxespad": 0.3,
        }
    )


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    width: float = TEXT_WIDTH,
    panel_aspect: float = 0.75,
    **kwargs,
):
    """plt.subplots at report size: `width` inches wide, each panel about panel_aspect times as
    tall as it is wide. Use width=TEXT_WIDTH / 2 for a single small plot."""
    height = nrows * panel_aspect * width / ncols
    return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)


def savefig(fig, path) -> list[Path]:
    """Write `fig` to <path>.pdf and <path>.png (200 dpi), creating parent directories.

    A .pdf or .png suffix on `path` is ignored; other dots are kept ("plateau_sigma0.01" gives
    "plateau_sigma0.01.pdf"). Returns the two paths.
    """
    path = Path(path)
    if path.suffix in (".pdf", ".png"):
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [path.with_name(path.name + ext) for ext in (".pdf", ".png")]
    # No creation date in the PDF, so regenerating an unchanged figure gives an identical file.
    fig.savefig(out[0], bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    fig.savefig(out[1], dpi=200, bbox_inches="tight", pad_inches=0.02)
    return out


# ---- helpers ---------------------------------------------------------------------------------


def _np(x) -> np.ndarray:
    """float64 array from numpy arrays, torch tensors or (nested) lists of them."""
    if hasattr(x, "detach"):  # torch tensor; duck-typed so that plotting does not import torch
        x = x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        x = [_np(e) for e in x]
    return np.asarray(x, dtype=np.float64)


def _positive(values: np.ndarray, log: bool) -> np.ndarray:
    """On a log axis, non-positive entries become NaN, drawn as a gap rather than clipped."""
    return np.where(values > 0, values, np.nan) if log else values


def _set_scales(ax, logx: bool, logy: bool) -> None:
    ax.set_xscale("log" if logx else "linear")
    ax.set_yscale("log" if logy else "linear")


def _default_labels(ax, xlabel: str, ylabel: str) -> None:
    """Axis labels, unless the caller has already set them."""
    if not ax.get_xlabel():
        ax.set_xlabel(xlabel)
    if not ax.get_ylabel():
        ax.set_ylabel(ylabel)


def _index_colors(n: int) -> list[str]:
    """Categorical slot per index; indices past the eighth are muted gray."""
    return list(PALETTE[:n]) + [MUTED] * max(0, n - len(PALETTE))


def _pm(value: float, err: float) -> str:
    """Mathtext "value \\pm err", both rounded at the first significant digit of err (at most 4
    decimals; 2 decimals if err is 0 or nan)."""
    decimals = int(np.clip(-np.floor(np.log10(err)), 0, 4)) if err > 0 else 2
    return rf"${value:.{decimals}f} \pm {err:.{decimals}f}$"


def _text_color_on(fill) -> str:
    """INK or white, whichever has the higher WCAG contrast with the fill colour."""
    c = np.asarray(to_rgb(fill))
    lum = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4) @ [0.2126, 0.7152, 0.0722]
    return "white" if 1.05 / (lum + 0.05) > (lum + 0.05) / 0.05 else INK


# ---- loss and overlaps -----------------------------------------------------------------------


def plot_loss(ax, t, loss, logx=True, logy=True, label=None, normalize=False, **kwargs):
    """Loss curve L(t), by default on log-log axes (plateaus at small init span decades in t).

    normalize=True plots L(t) / L(t_0), t_0 the first saved time. Points a log axis cannot show
    (t = 0, L <= 0) are left out. kwargs go to ax.plot; for one curve per value of an ordered
    parameter (sigma, chi, ...) pass color=ordinal_colors(n)[i]. Returns the Line2D.
    """
    t, loss = _np(t), _np(loss)
    if normalize:
        loss = loss / loss[0]
    (line,) = ax.plot(_positive(t, logx), _positive(loss, logy), label=label, **kwargs)
    _set_scales(ax, logx, logy)
    _default_labels(ax, "$t$", "$L(t)/L(0)$" if normalize else "$L(t)$")
    return line


def plot_loss_linear(ax, t, loss, label=None, **kwargs):
    """`plot_loss` on linear axes, which shows each loss drop at its true duration."""
    return plot_loss(ax, t, loss, logx=False, logy=False, label=label, **kwargs)


def plot_overlaps(
    ax, t, overlaps, labels=None, logx=True, logy=False, target=1.0, legend=True, **kwargs
):
    """CP component overlaps a_r(t), e.g. `CPTeacher.component_overlaps` at each saved time.

    overlaps: (T, R) array or a list of T length-R tensors. Component r takes categorical slot r
    (gray past the eighth) and legend label labels[r] (default $a_{r+1}$). logy=True plots
    |a_r(t)| on a log axis, which shows the growth during a plateau. A dashed hairline marks
    `target` (1 = component fully learned, for an odeco teacher and whitened inputs); None omits
    it. kwargs go to ax.plot. Returns the R lines.
    """
    t, a = _np(t), _np(overlaps)
    if logy:
        a = np.abs(a)
    if labels is None:
        labels = [f"$a_{{{r + 1}}}$" for r in range(a.shape[1])]
    x = _positive(t, logx)
    lines = [
        ax.plot(x, _positive(a[:, r], logy), color=color, label=labels[r], **kwargs)[0]
        for r, color in enumerate(_index_colors(a.shape[1]))
    ]
    if target is not None:
        ax.axhline(target, color=MUTED, lw=0.8, ls="--", zorder=1)
    _set_scales(ax, logx, logy)
    _default_labels(ax, "$t$", "$|a_r(t)|$" if logy else "$a_r(t)$")
    if legend and len(lines) > 1:
        ax.legend()
    return lines


# ---- bond spectra ----------------------------------------------------------------------------


def stack_spectra(spectra) -> dict[int, np.ndarray]:
    """{edge: (T, k) array} from a list of T dicts as returned by `bond_spectra` (one per saved
    time), or from a mapping that is already {edge: (T, k)}."""
    if isinstance(spectra, dict):
        return {v: _np(s) for v, s in spectra.items()}
    return {v: _np([s[v] for s in spectra]) for v in spectra[0]}


def _site_set(sites: list[int]) -> str:
    """Compact set notation; runs of three or more consecutive sites become "a–b"."""
    parts, i = [], 0
    while i < len(sites):
        j = i
        while j + 1 < len(sites) and sites[j + 1] == sites[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append(f"{sites[i]}–{sites[j]}")
        else:
            parts.extend(str(s) for s in sites[i : j + 1])
        i = j + 1
    return "{" + ",".join(parts) + "}"


def edge_label(tree, v: int) -> str:
    """The bipartition of the sites cut by edge v, sites below v first: "{0,1} | {2–7}"."""
    below = sorted(tree.subtree_sites[v])
    above = sorted(set(range(tree.n_sites)) - set(below))
    return f"{_site_set(below)} | {_site_set(above)}"


def _teacher_values(teacher, v: int, rel_floor: float | None) -> np.ndarray | None:
    if teacher is None:
        return None
    ref = np.sort(np.abs(_np(teacher[v] if isinstance(teacher, dict) else teacher)))[::-1]
    return ref[ref > (rel_floor or 0.0) * ref[0]]


def plot_bond_spectra(
    target,
    t,
    spectra,
    edges=None,
    tree=None,
    teacher=None,
    n_colored=None,
    rel_floor=1e-14,
    ncols=None,
    legend=True,
):
    """Singular values of W matricized across each edge, over time; one log-log panel per edge.

    target: a Figure (or SubFigure), on which a grid of panels with shared axes and `ncols` columns
    (default up to 4) is created, or a sequence of Axes, one per edge.
    spectra: a list of T dicts as returned by `bond_spectra`, or {edge: (T, k) array}.
    edges: the edges to show (default: all, in increasing order). tree: if given, each panel is
    titled with the bipartition its edge induces (`edge_label`), otherwise with the edge index.
    teacher: target singular values, drawn as dashed lines: one 1-D array for every edge (an odeco
    teacher's weights, which are its Schmidt values on every bond) or {edge: 1-D array} (e.g.
    `bond_spectra` of a TTN teacher). Absolute values, sorted decreasingly.
    Colours: the k-th largest singular value and the k-th teacher value share categorical slot k
    for k < n_colored (default: the teacher's rank on that edge, or every index without teacher,
    at most 8); the remaining indices are thinner muted-gray lines.
    Values below rel_floor x (the largest on that edge at that time) are hidden: `bond_spectra` is
    accurate to ~1e-15 relative, so they are rounding noise. None hides only zeros.
    Returns the list of Axes, in the order of `edges`.
    """
    data = stack_spectra(spectra)
    edges = sorted(data) if edges is None else list(edges)
    if isinstance(target, FigureBase):
        fig = target
        ncols = ncols or min(len(edges), 4)
        nrows = -(-len(edges) // ncols)
        grid = fig.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False).ravel()
        for ax in grid[len(edges) :]:
            ax.remove()
        axes = list(grid[: len(edges)])
        for i, ax in enumerate(axes):
            if i + ncols >= len(edges):  # no panel below this one, so it needs its own x ticks
                ax.tick_params(labelbottom=True)
        fig.supxlabel("$t$")
        fig.supylabel("singular value")
    else:
        axes = list(np.ravel(target))
        for ax in axes:
            ax.set_xlabel("$t$")
            ax.set_ylabel("singular value")

    x = _positive(_np(t), True)
    n_legend, any_gray = 0, False
    for ax, v in zip(axes, edges):
        s = data[v]
        s = np.where(s > (rel_floor or 0.0) * s.max(axis=1, keepdims=True), s, np.nan)
        ref = _teacher_values(teacher, v, rel_floor)
        n_ref = 0 if ref is None else len(ref)
        n_col = min(n_colored or (n_ref if ref is not None else s.shape[1]), len(PALETTE))
        n_all = max(s.shape[1], n_ref)
        colors = [PALETTE[k] if k < n_col else MUTED for k in range(n_all)]
        for k in range(s.shape[1]):
            ax.plot(x, s[:, k], color=colors[k], lw=1.5 if k < n_col else 0.9)
        for k in range(n_ref):
            ax.axhline(ref[k], color=colors[k], ls="--", lw=0.9, zorder=1.5)
        n_legend = max(n_legend, min(n_col, n_all))
        any_gray = any_gray or n_all > n_col
        ax.set_title(edge_label(tree, v) if tree is not None else f"edge {v}")
        _set_scales(ax, True, True)

    if legend:
        handles = [Line2D([], [], color=PALETTE[k], label=f"$k={k + 1}$") for k in range(n_legend)]
        if any_gray:
            handles.append(Line2D([], [], color=MUTED, lw=0.9, label="other $k$"))
        if teacher is not None:
            handles.append(Line2D([], [], color=MUTED, lw=0.9, ls="--", label="teacher"))
        if isinstance(target, FigureBase):
            fig.legend(handles=handles, loc="outside right upper")
        else:
            axes[0].legend(handles=handles)
    return axes


# ---- scaling ---------------------------------------------------------------------------------


def fit_power_law(x, y) -> tuple[float, float, float]:
    """Least-squares fit of log y = p log x + log c, i.e. y = c x^p.

    Returns (exponent p, prefactor c, standard error of p); the standard error is nan for two
    points.
    """
    res = linregress(np.log(_np(x)), np.log(_np(y)))
    return float(res.slope), float(np.exp(res.intercept)), float(res.stderr)


def plot_scaling(
    ax,
    x,
    y_mean,
    y_err=None,
    fit=None,
    fit_range=None,
    label=None,
    logx=True,
    logy=True,
    **kwargs,
):
    """Scaling plot, e.g. plateau time against init scale sigma: means with error bars and an
    optional least-squares fit, drawn over the fitted range.

    y_err: symmetric (n,) or asymmetric (2, n), as in ax.errorbar.
    fit: None; "power" for y = c x^p, fitted in log-log space (`fit_power_law`); or "log" for
    y = a ln x + b, fitted in semi-log space (the depth-2 law t ~ ln(1/sigma)); use logy=False to
    draw it as a straight line. The fitted exponent p (or slope a) and its standard error are
    appended to the legend label, and the legend is shown whenever there is a label (calling this
    once per series keeps every series in it). fit_range=(lo, hi) fits only the points with
    lo <= x <= hi (e.g. the small-sigma asymptotic regime).
    kwargs go to ax.errorbar (default: round markers with a white ring, no connecting line).
    Returns (p, c, stderr of p) for "power", (a, b, stderr of a) for "log", None without a fit.
    """
    x, y = _np(x), _np(y_mean)
    result, fit_line, fit_text = None, None, None
    if fit is not None:
        lo, hi = fit_range or (-np.inf, np.inf)
        sel = (x >= lo) & (x <= hi)
        xs = (np.geomspace if logx else np.linspace)(x[sel].min(), x[sel].max(), 100)
        if fit == "power":
            p, c, se = result = fit_power_law(x[sel], y[sel])
            fit_line = c * xs**p
            fit_text = f"exponent {_pm(p, se)}"
        elif fit == "log":
            res = linregress(np.log(x[sel]), y[sel])
            result = (float(res.slope), float(res.intercept), float(res.stderr))
            fit_line = res.slope * np.log(xs) + res.intercept
            fit_text = rf"slope vs $\ln x$: {_pm(res.slope, res.stderr)}"
        else:
            raise ValueError(f"fit must be None, 'power' or 'log', got {fit!r}")
        label = f"{label}, {fit_text}" if label else fit_text
    style = {"fmt": "o", "mec": SURFACE, "mew": 0.8, "elinewidth": 0.8} | kwargs
    bars = ax.errorbar(x, y, yerr=None if y_err is None else _np(y_err), label=label, **style)
    if fit_line is not None:
        ax.plot(xs, fit_line, color=bars.lines[0].get_color(), lw=1.0, zorder=1.9)
    _set_scales(ax, logx, logy)
    if label:
        ax.legend()
    return result


# ---- phase diagrams --------------------------------------------------------------------------


def _cells(values, log: bool):
    """(cell edges, cell centres, tick labels) for one axis of a phase diagram. Numbers sit at
    their values with edges at the (geometric, if log) midpoints; anything else gets evenly
    spaced cells labelled with str(value) (tick labels None for numbers)."""
    arr = np.asarray(values)
    if arr.dtype.kind not in "iuf":
        return (
            np.arange(len(arr) + 1) - 0.5,
            np.arange(len(arr), dtype=float),
            [str(v) for v in values],
        )
    centres = arr.astype(np.float64)
    u = np.log(centres) if log else centres
    half = np.diff(u) / 2 if len(u) > 1 else np.array([0.5])
    edges = np.concatenate([[u[0] - half[0]], u[:-1] + half, [u[-1] + half[-1]]])
    return (np.exp(edges) if log else edges), centres, None


def _missing(label) -> bool:
    return label is None or (isinstance(label, float) and np.isnan(label))


def plot_phase_diagram(
    ax,
    x_values,
    y_values,
    values,
    categorical=True,
    categories=None,
    colors=None,
    logx=False,
    logy=False,
    cmap=None,
    norm=None,
    colorbar_label=None,
    annotate=False,
    fmt="{:.2g}",
    legend=True,
):
    """Heatmap over a 2-D parameter grid: values[i][j] is the cell at (x_values[j], y_values[i]).

    categorical=True: values are regime labels; None (or NaN) marks a cell that was not computed
    and is left blank. Colours come from `colors` ({label: colour}) or from `regime_colors` applied
    to `categories` (default: the labels of `colors`, else the sorted labels present). Pass the same
    `categories` to every diagram so that a regime keeps its colour across figures. A legend to the
    right lists the regimes present.
    categorical=False: numbers (NaN = blank) on `cmap` (default `sequential_cmap()`) with a colorbar
    labelled `colorbar_label`; norm="log" (or any Normalize) for values spanning decades.
    Numeric axis values are placed at their values (log axis with logx/logy, cell edges at the
    geometric midpoints), so theory curves can be overlaid in data coordinates; non-numeric values
    (e.g. tree names; pass numbers as strings to get the same layout) become evenly spaced cells
    labelled with the values. Values must be sorted along each axis.
    annotate=True writes each cell's label, or its value formatted with `fmt`, inside the cell.
    Returns the QuadMesh.
    """
    x_edges, x_centres, x_ticks = _cells(x_values, logx)
    y_edges, y_centres, y_ticks = _cells(y_values, logy)
    if categorical:
        labels = np.asarray(values, dtype=object)
        present = {lab for lab in labels.ravel() if not _missing(lab)}
        if categories is None:
            categories = list(colors) if colors is not None else sorted(present, key=str)
        categories = list(categories)
        colors = colors if colors is not None else regime_colors(categories)
        codes = np.array(
            [[-1 if _missing(lab) else categories.index(lab) for lab in row] for row in labels]
        )
        z = np.ma.masked_less(codes, 0)
        n = len(categories)
        cmap = ListedColormap([colors[c] for c in categories])
        norm = BoundaryNorm(np.arange(n + 1) - 0.5, n)
        # Surface-coloured gaps keep neighbouring cells of similar colours apart.
        mesh = ax.pcolormesh(x_edges, y_edges, z, cmap=cmap, norm=norm, edgecolors=SURFACE, lw=0.6)
        if legend:
            handles = [Patch(color=colors[c], label=str(c)) for c in categories if c in present]
            ax.legend(
                handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0
            )
    else:
        z = np.ma.masked_invalid(_np(values))
        mesh = ax.pcolormesh(x_edges, y_edges, z, cmap=cmap or sequential_cmap(), norm=norm)
        ax.figure.colorbar(mesh, ax=ax, label=colorbar_label)

    ax.grid(False)
    _set_scales(ax, logx and x_ticks is None, logy and y_ticks is None)
    if x_ticks is not None:
        ax.set_xticks(x_centres, labels=x_ticks, minor=False)
    if y_ticks is not None:
        ax.set_yticks(y_centres, labels=y_ticks, minor=False)
    if annotate:
        mask = np.ma.getmaskarray(z)
        for i, yc in enumerate(y_centres):
            for j, xc in enumerate(x_centres):
                if mask[i, j]:
                    continue
                text = str(labels[i, j]) if categorical else fmt.format(z[i, j])
                fill = mesh.cmap(mesh.norm(z[i, j]))
                ax.text(
                    xc, yc, text, ha="center", va="center", fontsize=7, color=_text_color_on(fill)
                )
    return mesh
