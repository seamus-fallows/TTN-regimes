"""Run configurations: nested dataclasses, loadable from YAML, with dotted-key overrides.

A run is fully described by a `RunConfig` (plus the code version). YAML files may give any subset
of the fields; missing ones take the defaults below, unknown keys are errors. Values are coerced to
the declared field types, so YAML's `1e-8` (a string in YAML 1.1) is read as the float 1e-8, and
choice fields (Literal types) reject unknown values when the config is built, before anything runs.
Files are read and written as UTF-8 on every platform.

    cfg = RunConfig.load("experiments/00_smoke/configs/base.yaml", ["init.sigma=0.01"])
    cfg.with_overrides({"optimizer.method": "lsoda", "chi": [3, 3]})

Override values are parsed as YAML: `chi=[2,4]`, `optimizer.loss_floor=null`, `tree.kind=mps`.

Seeding: every run has one integer `seed`. The runner derives independent random streams from it
with fixed stream offsets (`stream_seed`): the teacher seed (unless `teacher.seed` is set, e.g. to
keep one teacher across init seeds), the init generator, the data generator (training set, SGD
batches) and the probe generator (tangent-kernel probe inputs).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

TreeKind = Literal["binary", "mps", "ht"]


@dataclass
class TreeConfig:
    kind: TreeKind = "binary"  # `ttn.tree.make_tree`
    n_sites: int = 4


@dataclass
class InitConfig:
    """scheme:
    fanin          G_v = sigma_v Z_v / sqrt(F_v) (`TTNSpec.init`), sigma_v = sigma * core_scales[v]
    aligned        cores diagonal in a CP teacher's basis (`aligned_init`) with per-core,
                   per-component amplitudes aligned_a0: a scalar, a list over components, or a
                   (K, R) nested list; null -> sigma * core_scales[v] on every component
    antisymmetric  two-copy model W(theta+) - W(theta-), theta+ = theta- = fan-in init
                   (`antisymmetric_double` + gradient mask), so W_0 = 0 at any sigma
    centered       centred model: fan-in init, target W* + W_0 (`ShiftedTeacher`); "centred" is
                   accepted and stored as "centered"
    """

    scheme: Literal["fanin", "aligned", "antisymmetric", "centered", "centred"] = "fanin"
    sigma: float = 0.05
    core_scales: list[float] | None = None  # one per core (node order); null = all 1
    aligned_a0: float | list | None = None

    def __post_init__(self):
        if self.scheme == "centred":
            self.scheme = "centered"


@dataclass
class TeacherConfig:
    """kind: odeco | pattern | random_cp (CP teachers with `weights`) or random_ttn (Gaussian
    TTN on tree `ttn_tree_kind` (null = the student's) with bond dimension `ttn_chi`, scaled to
    Frobenius norm `ttn_norm`). patterns: per-component site patterns of `pattern_teacher`.
    basis: random (Haar) or standard site bases (odeco, pattern). seed: null = derived from the
    run seed."""

    kind: Literal["odeco", "pattern", "random_cp", "random_ttn"] = "odeco"
    weights: list[float] = field(default_factory=lambda: [3.0, 2.0, 1.0])
    patterns: list[list[int]] | None = None
    seed: int | None = None
    basis: Literal["random", "standard"] = "random"
    ttn_tree_kind: TreeKind | None = None
    ttn_chi: int = 2
    ttn_norm: float = 1.0


@dataclass
class InputsConfig:
    """Site covariances Sigma_i: identity (whitened); power_law (`power_law_cov(d, exponent)` at
    every site); custom (`matrices`: one d x d matrix for every site, or a list of N of them)."""

    cov: Literal["identity", "power_law", "custom"] = "identity"
    exponent: float = 1.0
    matrices: list | None = None


@dataclass
class LossConfig:
    """population: exact population loss. empirical: n_train fixed samples whose labels carry
    N(0, label_noise^2) noise. For online SGD (sgd + population) label_noise is fresh noise on
    every batch; for gf / gd with the population loss it has no effect."""

    kind: Literal["population", "empirical"] = "population"
    n_train: int = 1000
    label_noise: float = 0.0


@dataclass
class OptimizerConfig:
    """kind gf: gradient flow up to t_max, method dopri5 (reference) | lsoda (stiff later
    plateaus) | radau, tolerances rtol / atol, at most max_steps steps. kind gd / sgd: n_steps
    steps of size lr (sgd: minibatches of batch_size). Every kind stops once the training loss is
    below loss_floor (null = never), and after max_wall_time seconds of training (null = never)."""

    kind: Literal["gf", "gd", "sgd"] = "gf"
    method: Literal["dopri5", "lsoda", "radau"] = "dopri5"
    t_max: float = 1e4
    rtol: float = 1e-8
    atol: float = 1e-12
    max_steps: int = 10_000_000
    lr: float = 0.01
    n_steps: int = 10_000
    batch_size: int = 32
    loss_floor: float | None = 1e-10
    max_wall_time: float | None = None


@dataclass
class LoggingConfig:
    """n_log log-spaced log times (plus t = 0 and the end; gf: `ttn.dynamics.default_log_times`).
    log_times: explicit log times instead (gd / sgd: rounded to steps of size lr), e.g. one grid
    shared by runs that are averaged. fine / fine_tol: fine rows at the integrator steps where the
    state changed (`ttn.dynamics.Recorder`), which resolve learning events; `summary` needs them for
    event widths and separation ratios. spectra: record bond spectra. save_params: store theta at
    every log time. ntk_probe: number of probe inputs for the tangent kernel (0 = off): records
    ntk_change (delta_K) and ntk_alignment at every log time. hessian_at: times at which the full
    and gauge-projected Hessian spectra are recorded."""

    n_log: int = 200
    log_times: list[float] | None = None
    fine: bool = True
    fine_tol: float = 0.01
    save_params: bool = False
    spectra: bool = True
    ntk_probe: int = 0
    hessian_at: list[float] = field(default_factory=list)


@dataclass
class RunConfig:
    """One training run. chi: one bond dimension for every edge, or a list in `tree.edges` order
    (non-root nodes in increasing order); cap_bonds caps each at the rank its bipartition allows.
    lr_multipliers: per-core learning-rate multipliers (node order), null = all 1."""

    name: str = "run"
    seed: int = 0
    tree: TreeConfig = field(default_factory=TreeConfig)
    d: int = 3
    chi: int | list[int] = 3
    cap_bonds: bool = True
    init: InitConfig = field(default_factory=InitConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    inputs: InputsConfig = field(default_factory=InputsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    lr_multipliers: list[float] | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> RunConfig:
        return _from_dict(cls, data or {}, "")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, path: str | Path, overrides=()) -> RunConfig:
        """Read a YAML file and apply overrides ("key=value" strings or a {key: value} dict)."""
        return cls.from_dict(load_yaml(path)).with_overrides(overrides)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def with_overrides(self, overrides) -> RunConfig:
        """A new config with dotted keys set: ["init.sigma=0.01", ...] or {"init.sigma": 0.01}."""
        if not isinstance(overrides, dict):
            overrides = dict(parse_override(item) for item in overrides)
        data = self.to_dict()
        for key, value in overrides.items():
            *parents, last = key.split(".")
            node = data
            for part in parents:
                if not isinstance(node.get(part), dict):
                    raise KeyError(f"unknown config key {key!r}")
                node = node[part]
            if last not in node:
                raise KeyError(f"unknown config key {key!r}")
            node[last] = value
        return RunConfig.from_dict(data)

    def get(self, key: str):
        """Value at a dotted key, e.g. cfg.get("init.sigma")."""
        value = self
        for part in key.split("."):
            value = getattr(value, part)
        return value


def flatten_dict(data: dict, prefix: str = "") -> dict:
    """{"init": {"sigma": 0.1}} -> {"init.sigma": 0.1} (lists are kept as values)."""
    out = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out.update(flatten_dict(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def parse_override(item: str) -> tuple[str, object]:
    """ "init.sigma=0.01" -> ("init.sigma", 0.01); the value is parsed as YAML."""
    key, sep, value = item.partition("=")
    if not sep:
        raise ValueError(f"override {item!r} is not of the form key=value")
    return key.strip(), yaml.safe_load(value)


def load_yaml(path: str | Path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# Fixed offsets of the random streams of one run (never Python's hash(): it is salted per process).
SEED_STREAMS = {"teacher": 1, "init": 2, "data": 3, "probe": 4}


def stream_seed(seed: int, stream: str) -> int:
    """Seed of one random stream of a run: a 32-bit hash of (run seed, stream offset) by numpy's
    SeedSequence, so streams of different runs and different streams never coincide."""
    state = np.random.SeedSequence([seed, SEED_STREAMS[stream]]).generate_state(1)
    return int(state[0])


# ---- dict -> dataclass with type coercion --------------------------------------------------------


def _from_dict(cls, data: dict, where: str):
    """`where`: dotted path of this node ("" at the top), for error messages."""
    if not isinstance(data, dict):
        raise TypeError(f"{where or 'config'}: expected a mapping, got {data!r}")
    hints = typing.get_type_hints(cls)
    unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        keys = sorted(f"{where}.{k}" if where else k for k in unknown)
        raise KeyError(f"unknown config keys {keys}")
    kwargs = {
        key: _coerce(value, hints[key], f"{where}.{key}" if where else key)
        for key, value in data.items()
    }
    return cls(**kwargs)


def _coerce(value, hint, where: str):
    """Convert a YAML value to the declared type: nested dataclasses, X | None, scalar | list,
    list[X], float (also from strings such as "1e-8" and from ints), int (also from "1e4"),
    Literal choices (the value must be one of them)."""
    if dataclasses.is_dataclass(hint):
        return _from_dict(hint, value, where)
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is Literal:
        if value not in args:
            raise ValueError(f"{where}: {value!r} is not one of {list(args)}")
        return value
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        options = [a for a in args if a is not type(None)]
        is_list = [a is list or typing.get_origin(a) is list for a in options]
        if len(options) > 1:  # scalar-or-list fields: pick the branch by the value's shape
            options = [a for a, lst in zip(options, is_list) if lst == isinstance(value, list)]
        return _coerce(value, options[0], where)
    if hint is list or origin is list:
        if not isinstance(value, list | tuple):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        inner = args[0] if args else None
        return [v if inner is None else _coerce(v, inner, where) for v in value]
    if hint is float:
        return float(value)
    if hint is int:
        number = float(value) if isinstance(value, str) else value
        if isinstance(number, float):
            if not number.is_integer():
                raise ValueError(f"{where}: expected an integer, got {value!r}")
            number = int(number)
        return number
    if hint is bool and not isinstance(value, bool):
        raise TypeError(f"{where}: expected true or false, got {value!r}")
    return value
