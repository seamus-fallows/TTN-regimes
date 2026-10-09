# Training dynamics of tree tensor networks

Which training regimes do tree tensor networks (TTNs) trained by gradient methods show (in
particular saddle-to-saddle / incremental learning), and how do initialization scale, bond
dimensions, tree shape, input covariance, teacher structure, learning rate and overparameterization
select them?

- Project map, conventions and status: [CLAUDE.md](CLAUDE.md)
- Research plan: [PLAN.md](PLAN.md)
- The model: [notes/model.md](notes/model.md)
- Report: [report/](report/)

## Setup

```
uv sync
uv run pytest
```

## Data

Small summary data and figures are committed under `experiments/<name>/results/`. Large raw outputs
are kept outside git in `data/<name>/` (gitignored); each experiment's README says how to regenerate
them.
