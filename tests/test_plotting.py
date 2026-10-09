from itertools import pairwise

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.colors import to_hex, to_rgb
from PIL import Image

from ttn.network import TTNSpec, aligned_init, bond_spectra
from ttn.plotting import (
    MUTED,
    PALETTE,
    REGIME_PALETTE,
    TEXT_WIDTH,
    edge_label,
    fit_power_law,
    ordinal_colors,
    palette,
    plot_bond_spectra,
    plot_loss,
    plot_loss_linear,
    plot_overlaps,
    plot_phase_diagram,
    plot_scaling,
    regime_colors,
    savefig,
    set_style,
    stack_spectra,
    subplots,
)
from ttn.teachers import odeco_teacher
from ttn.tree import binary_tree, mps_tree

mpl.use("Agg")


@pytest.fixture(autouse=True)
def style():
    set_style()
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def single_torch_thread():
    # The tensors here are tiny; with torch's default thread pool on a machine busy with parallel
    # experiments, spin-waiting threads made these tests ~400x slower.
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def test_style_and_palettes():
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == list(PALETTE)
    assert mpl.rcParams["image.cmap"] == "ttn_blue" and mpl.rcParams["pdf.fonttype"] == 42
    assert palette() == list(PALETTE) and palette(3) == list(PALETTE[:3])
    with pytest.raises(ValueError):
        palette(9)
    brightness = [np.mean(to_rgb(c)) for c in ordinal_colors(7)]
    assert all(a > b for a, b in pairwise(brightness))  # light -> dark
    assert list(regime_colors(["s2s", "lazy", "rich"]).values()) == list(REGIME_PALETTE[:3])
    assert list(regime_colors("abcdef").values()) == list(PALETTE[:6])
    set_style("sans")
    assert mpl.rcParams["font.family"] == ["DejaVu Sans"]


def test_fit_power_law_recovers_exponent():
    x = np.geomspace(1e-3, 1.0, 12)
    p, c, se = fit_power_law(x, 2.5 * x**-1.7)
    assert p == pytest.approx(-1.7, abs=1e-12) and c == pytest.approx(2.5, rel=1e-12)
    assert se < 1e-6  # linregress gets the error through 1 - r^2, so ~1e-8 is its floor
    y = 2.5 * x**-1.7 * np.exp(0.05 * np.random.default_rng(0).standard_normal(x.size))
    p, c, se = fit_power_law(torch.tensor(x), torch.tensor(y))
    assert 0 < se < 0.02 and abs(p + 1.7) < 4 * se and c == pytest.approx(2.5, rel=0.1)


def test_plot_loss(tmp_path):
    t = np.linspace(0.0, 50.0, 200)
    loss = 0.5 * np.exp(-t / 5) + 1e-8
    fig, (ax1, ax2) = subplots(1, 2)
    line = plot_loss(ax1, t, loss, label="a", normalize=True)
    assert np.isnan(line.get_xdata()[0])  # t = 0 has no place on a log axis
    assert line.get_ydata()[0] == 1.0
    assert ax1.get_xscale() == ax1.get_yscale() == "log" and ax1.get_ylabel() == "$L(t)/L(0)$"
    plot_loss(ax1, t, 2 * torch.tensor(loss), color=ordinal_colors(2)[1], label="b")
    line = plot_loss_linear(ax2, list(t), [torch.tensor(v) for v in loss])
    assert ax2.get_xscale() == ax2.get_yscale() == "linear"
    np.testing.assert_allclose(line.get_ydata(), loss)
    pdf, png = savefig(fig, tmp_path / "loss")
    assert pdf.stat().st_size > 0 and png.stat().st_size > 0


def test_plot_overlaps(tmp_path):
    n, d, weights = 4, 3, [3.0, 2.0, 1.0]
    teacher = odeco_teacher(n, d, weights)
    spec = TTNSpec.uniform(binary_tree(n), d, 3)
    t = np.concatenate([[0.0], np.geomspace(1e-2, 1e2, 40)])
    overlaps = []
    for tk in t:
        # Saxe-like decoupled trajectory: component r learned around t = 3 / s_r.
        amps = torch.ones(spec.n_cores, 3, dtype=torch.float64)
        a = torch.tensor([sigmoid(5 * (tk * w / 3 - 1)) for w in weights], dtype=torch.float64)
        amps[spec.tree.root] = a * torch.tensor(weights, dtype=torch.float64)
        overlaps.append(teacher.component_overlaps(spec, aligned_init(spec, teacher.factors, amps)))
    fig, (ax1, ax2) = subplots(1, 2)
    lines = plot_overlaps(ax1, t, overlaps, labels=[f"$s={w:g}$" for w in weights])
    assert [line.get_color() for line in lines] == palette(3)
    np.testing.assert_allclose(lines[1].get_ydata(), [o[1].item() for o in overlaps], atol=1e-12)
    assert [x.get_text() for x in ax1.get_legend().get_texts()] == ["$s=3$", "$s=2$", "$s=1$"]
    lines = plot_overlaps(ax2, t, -np.stack([o.numpy() for o in overlaps]), logy=True)
    assert ax2.get_yscale() == "log" and np.nanmin(lines[0].get_ydata()) > 0
    savefig(fig, tmp_path / "overlaps")


def test_plot_bond_spectra(tmp_path):
    n, d, weights = 8, 3, [2.0, 0.5]
    teacher = odeco_teacher(n, d, weights)
    spec = TTNSpec.uniform(binary_tree(n), d, 3)
    noise = spec.init(1.0, seed=1)
    t = np.concatenate([[0.0], np.geomspace(1e-2, 1e2, 30)])
    spectra = []
    for tk in t:
        amps = torch.ones(spec.n_cores, 2, dtype=torch.float64)
        a = torch.tensor([sigmoid(5 * (tk * w - 1)) for w in weights], dtype=torch.float64)
        amps[spec.tree.root] = a * torch.tensor(weights, dtype=torch.float64)
        cores = aligned_init(spec, teacher.factors, amps)
        spectra.append(bond_spectra(spec, [c + 1e-3 * z for c, z in zip(cores, noise)]))

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.0))
    axes = plot_bond_spectra(fig, t, spectra, tree=spec.tree, teacher=teacher.weights)
    assert len(axes) == len(spec.tree.edges) == 6 and len(fig.axes) == 6
    assert axes[0].get_xscale() == axes[0].get_yscale() == "log"
    assert axes[2].get_title(loc="left") == edge_label(spec.tree, 2) == "{0–3} | {4–7}"
    # Student lines (k = 1, 2 coloured, k = 3 gray) then the two dashed teacher lines.
    lines = axes[0].lines
    assert [line.get_color() for line in lines] == [*palette(2), MUTED, *palette(2)]
    assert [line.get_linestyle() for line in lines[3:]] == ["--", "--"]
    np.testing.assert_allclose(lines[3].get_ydata(), [2.0, 2.0])
    np.testing.assert_allclose(lines[0].get_ydata()[1:], stack_spectra(spectra)[0][1:, 0])
    texts = [x.get_text() for x in fig.legends[0].get_texts()]
    assert texts == ["$k=1$", "$k=2$", "other $k$", "teacher"]
    savefig(fig, tmp_path / "spectra")

    # A selection of edges on given axes, with per-edge teacher spectra.
    amps = torch.ones(spec.n_cores, 2, dtype=torch.float64)
    amps[spec.tree.root] = torch.tensor(weights, dtype=torch.float64)
    target = bond_spectra(spec, aligned_init(spec, teacher.factors, amps))
    fig, axs = subplots(1, 2)
    axes = plot_bond_spectra(axs, t, stack_spectra(spectra), edges=[0, 3], teacher=target)
    assert [ax.get_title(loc="left") for ax in axes] == ["edge 0", "edge 3"]
    assert axes[0].get_legend() is not None
    # The teacher's third Schmidt value is ~1e-16, hidden as rounding noise.
    assert len([line for line in axes[1].lines if line.get_linestyle() == "--"]) == 2
    savefig(fig, tmp_path / "spectra_axes")


def test_plot_scaling(tmp_path):
    sigma = np.geomspace(1e-3, 1e-1, 8)
    rng = np.random.default_rng(1)
    runs = 3.0 * sigma**-2.0 * np.exp(0.03 * rng.standard_normal((5, 8)))
    fig, (ax1, ax2) = subplots(1, 2)
    p, _, se = plot_scaling(ax1, sigma, runs.mean(0), runs.std(0), fit="power", label="binary8")
    assert abs(p + 2.0) < 0.05 and se < 0.05
    plot_scaling(ax1, sigma, 2 * runs.mean(0), fit="power", label="mps8")
    texts = [x.get_text() for x in ax1.get_legend().get_texts()]
    assert len(texts) == 2 and texts[0].startswith("binary8, exponent $")
    # "exponent $p \pm se$", both rounded at the first significant digit of se
    shown_p, _, shown_se = texts[0].split("$")[1].split()
    assert float(shown_p) == pytest.approx(p, abs=se) and float(shown_se) == pytest.approx(
        se, rel=0.5
    )

    y = -4.0 * np.log(sigma) + 1.0
    y[-1] += 10.0  # outside the fitted range
    a, b, _ = plot_scaling(ax2, sigma, y, fit="log", fit_range=(1e-3, 3e-2), logy=False)
    assert a == pytest.approx(-4.0) and b == pytest.approx(1.0)
    assert ax2.lines[-1].get_xdata().max() <= 3e-2  # fit drawn over the fitted range only
    with pytest.raises(ValueError):
        plot_scaling(ax2, sigma, y, fit="exp")
    savefig(fig, tmp_path / "scaling")


def test_plot_phase_diagram(tmp_path):
    sigma = np.geomspace(1e-3, 1.0, 4)
    trees = ["binary8", "mps8"]
    labels = [["s2s", "s2s", "rich", "lazy"], ["s2s", None, "lazy", "lazy"]]
    regimes = ["s2s", "rich", "lazy", "diverged"]
    fig, (ax1, ax2) = subplots(1, 2)
    mesh = plot_phase_diagram(
        ax1, sigma, trees, labels, logx=True, categories=regimes, annotate=True
    )
    assert [x.get_text() for x in ax1.get_legend().get_texts()] == ["s2s", "rich", "lazy"]
    assert [x.get_text() for x in ax1.get_yticklabels()] == trees
    assert ax1.get_xscale() == "log" and len(ax1.texts) == 7
    assert np.ma.getmaskarray(mesh.get_array()).sum() == 1
    # Colours follow the order of `categories`, not which regimes happen to be present.
    assert to_hex(mesh.cmap(mesh.norm(2))) == regime_colors(regimes)["lazy"] == REGIME_PALETTE[2]

    values = np.array([[1.0, 10.0, np.nan], [100.0, 1e3, 1e4]])
    mesh = plot_phase_diagram(
        ax2,
        [2, 4, 8],
        [0.1, 0.2],
        values,
        categorical=False,
        norm="log",
        colorbar_label="plateau time",
        annotate=True,
    )
    assert mesh.colorbar.ax.get_ylabel() == "plateau time"
    assert len(ax2.texts) == 5
    savefig(fig, tmp_path / "phase")


def test_savefig_paths_and_resolution(tmp_path):
    fig, ax = subplots(width=TEXT_WIDTH / 2)
    ax.plot([1.0, 2.0], [1.0, 3.0])
    pdf, png = savefig(fig, tmp_path / "a" / "b" / "plateau_sigma0.01")
    assert (pdf.name, png.name) == ("plateau_sigma0.01.pdf", "plateau_sigma0.01.png")
    assert pdf.read_bytes().startswith(b"%PDF")
    with Image.open(png) as image:
        assert image.info["dpi"] == pytest.approx((200, 200), abs=0.5)
    again, _ = savefig(fig, tmp_path / "again.pdf")
    assert again == tmp_path / "again.pdf"
    assert again.read_bytes() == pdf.read_bytes()  # no timestamp: regenerated files are identical


def test_edge_label_and_stack_spectra():
    tree = binary_tree(8)
    assert edge_label(tree, 0) == "{0,1} | {2–7}"
    assert edge_label(mps_tree(5), 2) == "{0–2} | {3,4}"
    spec = TTNSpec.uniform(tree, 2, 2)
    per_time = [bond_spectra(spec, spec.init(1.0, seed=s)) for s in range(3)]
    stacked = stack_spectra(per_time)
    assert sorted(stacked) == list(tree.edges) and stacked[0].shape == (3, 2)
    np.testing.assert_array_equal(stacked[4][2], per_time[2][4].numpy())
