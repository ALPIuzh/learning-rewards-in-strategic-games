"""Render reviewable figure alternatives without changing approved figures."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from reproducibility.figures.manifest import REPOSITORY_ROOT
from reproducibility.figures.recipes.lobeke import CURVES, coordinates
from reproducibility.figures.recipes.lobeke_diagnostics import METHODS as LOBEKE_METHODS
from reproducibility.figures.recipes.lobeke_diagnostics import _curve as lobeke_curve
from reproducibility.figures.recipes.structured_temperature import ETA_VALUES
from reproducibility.figures.recipes.structured_warmup import METHODS as STRUCTURED_METHODS
from reproducibility.figures.recipes.structured_warmup import WARMUP_BUDGETS
from reproducibility.figures.recipes.zero_sum import (
    _downstream_plots,
    _source_plots,
)
from reproducibility.figures.statistics import mean_interval

OUTPUT_DIR = REPOSITORY_ROOT / "reproducibility" / "alternatives" / "generated"


def _tuning_interval(statistics: Mapping[str, object]) -> tuple[float, float]:
    mean = float(statistics["mean"])
    standard_error = float(statistics["standard_error"])
    seeds = int(statistics["seeds"])
    critical = float(student_t.ppf(0.975, df=seeds - 1))
    return mean, critical * standard_error


def tuning_coordinates(
    candidates: Iterable[Mapping[str, object]], *, scale: float
) -> str:
    points = []
    for candidate in candidates:
        mean, radius = _tuning_interval(candidate["stackelberg_suboptimality"])
        multiplier = float(candidate["lambda_information_multiplier"])
        points.append(f"({multiplier:.8g},{scale * mean:.8g}) +- (0,{scale * radius:.8g})")
    return " ".join(points)


def _selected_coordinate(
    candidates: Iterable[Mapping[str, object]],
    selected: Mapping[str, object],
    *,
    scale: float,
) -> str:
    key = (int(selected["warmup_budget"]), float(selected["lambda_information_multiplier"]))
    for candidate in candidates:
        candidate_key = (
            int(candidate["warmup_budget"]),
            float(candidate["lambda_information_multiplier"]),
        )
        if candidate_key == key:
            mean, _ = _tuning_interval(candidate["stackelberg_suboptimality"])
            return f"({key[1]:.8g},{scale * mean:.8g})"
    raise ValueError("selected candidate is absent from the historical alternative")


def _effect_coordinates(
    confirmation: Iterable[Mapping[str, object]], metric: str, *, scale: float
) -> str:
    by_mode = {str(item["mode"]): item for item in confirmation}
    points = []
    for x_value, mode in ((1, "warmup"), (2, "no_warmup")):
        mean, radius = _tuning_interval(by_mode[mode]["paired_differences"][metric])
        points.append(f"({x_value},{scale * mean:.8g}) +- (0,{scale * radius:.8g})")
    return " ".join(points)


def _final_relation_points(arrays: np.lib.npyio.NpzFile) -> str:
    commands: list[str] = []
    expert_methods = {"fixed_expert_mle", "zero_sum_ols", "follower_ols"}
    for slug, key, _label in LOBEKE_METHODS:
        fisher = arrays[f"{slug}__target_fisher_error"][:, -1:]
        subopt = arrays[f"{slug}__target_suboptimality"][:, -1:]
        fisher_mean, fisher_low, fisher_high = mean_interval(fisher, 0.95)
        subopt_mean, subopt_low, subopt_high = mean_interval(subopt, 0.95)
        x_radius = max(fisher_mean[0] - fisher_low[0], fisher_high[0] - fisher_mean[0])
        y_radius = max(subopt_mean[0] - subopt_low[0], subopt_high[0] - subopt_mean[0])
        color, _line_style, _ = CURVES[key]
        if slug in expert_methods:
            mark, fill = "square*", color
        elif key[1]:
            mark, fill = "*", color
        else:
            mark, fill = "o", "white"
        commands.append(
            rf"\addplot[only marks, mark={mark}, mark size=1.8pt, draw={color}, "
            rf"fill={fill}, line width=0.55pt, forget plot, error bars/.cd, x dir=both, "
            rf"x explicit, y dir=both, y explicit] coordinates "
            rf"{{({fisher_mean[0]:.8g},{subopt_mean[0]:.8g}) +- "
            rf"({x_radius:.8g},{y_radius:.8g})}};"
        )
    return "\n".join(commands)


def render_lobeke_relation() -> str:
    """Show learning curves once, then relate final Fisher error to deployment."""

    data_path = REPOSITORY_ROOT / "reproducibility/data/lobeke/final_evaluation/data.npz"
    with np.load(data_path, allow_pickle=False) as arrays:
        curves = "\n".join(
            lobeke_curve(
                arrays,
                slug=slug,
                key=key,
                label=label,
                metric="target_suboptimality",
                floor=3.0e-5,
                add_legend=False,
            )
            for slug, key, label in LOBEKE_METHODS
        )
        relation = _final_relation_points(arrays)
    return rf"""% Exploratory alternative; generated from protected Lobeke data.
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 1, group name=lobekerel, horizontal sep=1.55cm}},
  mairl axis,
  width=0.458\textwidth,
  height=5.40cm,
  ytick style={{draw=none}},
  minor y tick style={{draw=none}}
]
\nextgroupplot[
  title={{(a) Target SubOpt}},
  xmin=0, xmax=1600,
  xtick={{0,400,800,1200,1600}},
  xlabel={{Number of observed poacher actions}},
  ylabel={{${{\operatorname{{SubOpt}}(\hat\mu)}}$}},
  ymode=log,
  ymin=3e-5, ymax=0.15
]
\path[fill=mairlActive!5] (axis cs:0,3e-5) rectangle (axis cs:200,0.15);
\draw[mairlGrid, densely dotted, line width=0.45pt]
  (axis cs:200,3e-5) -- (axis cs:200,0.15);
\node[rotate=90, font=\tiny, text=mairlActive, anchor=center]
  at (axis cs:100,1.1e-4) {{Warm-up}};
{curves}
\nextgroupplot[
  title={{(b) Target Fisher--SubOpt relation}},
  xlabel={{Final Target Fisher error}},
  ylabel={{Final ${{\operatorname{{SubOpt}}(\hat\mu)}}$}},
  xmode=log,
  ymode=log,
  xmin=2e-2, xmax=1.25,
  ymin=4e-5, ymax=5e-2
]
{relation}
\end{{groupplot}}
\node[anchor=north, font=\scriptsize, text=mairlInk, align=left]
  at ($(lobekerel c1r1.south)!0.5!(lobekerel c2r1.south)+(0,-1.38cm)$) {{
  \begin{{tabular}}{{@{{}}lll@{{\hspace{{8pt}}}}lll@{{}}}}
  \textcolor{{mairlActive}}{{\rule[0.4ex]{{0.24cm}}{{0.55pt}}}}\ Active IRL &
  \textcolor{{mairlFixed}}{{\rule[0.4ex]{{0.24cm}}{{0.55pt}}}}\ Fixed random &
  \textcolor{{mairlOnline}}{{\rule[0.4ex]{{0.24cm}}{{0.55pt}}}}\ Online random &
  $\bullet$\ w/ warm-up & $\circ$\ w/o warm-up & \\
  \textcolor{{mairlExpert}}{{$\blacksquare$}}\ Fixed-expert MLE &
  \textcolor{{mairlZeroSum}}{{$\blacksquare$}}\ Zero-sum OLS &
  \textcolor{{mairlFollower}}{{$\blacksquare$}}\ Follower-only OLS & & &
  \end{{tabular}}
  }};
\end{{tikzpicture}}
"""


def _warm_selection_lines(candidates: tuple[dict[str, Any], ...]) -> str:
    budgets = sorted({int(candidate["warmup_budget"]) for candidate in candidates})
    styles = (
        ("mairlActive!45!mairlInk", "*"),
        ("mairlActive!60!mairlInk", "square*"),
        ("mairlActive!75!mairlInk", "triangle*"),
        ("mairlActive", "diamond*"),
    )
    commands: list[str] = []
    for budget, (color, mark) in zip(budgets, styles, strict=True):
        group = sorted(
            (candidate for candidate in candidates if int(candidate["warmup_budget"]) == budget),
            key=lambda item: float(item["lambda_information_multiplier"]),
        )
        points = " ".join(
            f"({float(item['lambda_information_multiplier']):.8g},"
            f"{1.0e4 * float(item['stackelberg_suboptimality']['mean']):.8g})"
            for item in group
        )
        commands.extend(
            (
                rf"\addplot[draw={color}, mark={mark}, mark size=1.55pt, "
                rf"line width=0.55pt] coordinates {{{points}}};",
                rf"\addlegendentry{{$K_0={budget}$}}",
            )
        )
    return "\n".join(commands)


def render_tuning_spacious() -> str:
    """Give the warm-up search most of the area and stack secondary checks."""

    path = REPOSITORY_ROOT / "reproducibility/data/lobeke/tuning.json"
    payload = json.loads(path.read_text())
    warm = payload["selection"]["warmup"]
    no_warm = payload["selection"]["no_warmup"]
    warm_candidates = tuple(warm["candidates"])
    no_warm_candidates = tuple(no_warm["candidates"])
    warm_lines = _warm_selection_lines(warm_candidates)
    warm_selected = _selected_coordinate(warm_candidates, warm["selected"], scale=1.0e4)
    no_warm_coordinates = tuning_coordinates(no_warm_candidates, scale=1.0e4)
    no_warm_selected = _selected_coordinate(
        no_warm_candidates,
        no_warm["selected"],
        scale=1.0e4,
    )
    effects = _effect_coordinates(
        payload["confirmation"],
        "stackelberg_suboptimality",
        scale=1.0e4,
    )
    return rf"""% Exploratory alternative; generated from protected tuning data.
\begin{{tikzpicture}}
\begin{{axis}}[
  name=warmselect,
  mairl axis,
  width=0.59\textwidth,
  height=6.45cm,
  title={{(a) Joint selection with warm-up}},
  xmode=log,
  log basis x=2,
  xmin=0.85, xmax=38,
  xtick={{1,2,4,8,16,32}},
  xticklabels={{1,2,4,8,16,32}},
  xlabel={{Information multiplier $\lambda/\eta^2$}},
  ylabel={{Final $\operatorname{{SubOpt}}(\hat\mu)$ ($10^{{-4}}$)}},
  ymin=0.75, ymax=3.15,
  legend style={{at={{(0.03,0.97)}}, anchor=north west, legend columns=2,
                 font=\tiny, column sep=4pt, row sep=-1pt}}
]
{warm_lines}
\addplot[only marks, mark=o, mark size=2.9pt, line width=0.8pt,
         draw=mairlInk, forget plot] coordinates {{{warm_selected}}};
\node[anchor=south east, font=\tiny, text=mairlMuted]
  at (rel axis cs:0.98,0.03) {{ring: selected}};
\end{{axis}}
\begin{{axis}}[
  at={{($(warmselect.south east)+(1.12cm,3.35cm)$)}},
  anchor=south west,
  name=nowarmselect,
  mairl axis,
  width=0.34\textwidth,
  height=3.18cm,
  title={{(b) No warm-up selection}},
  xmode=log,
  log basis x=2,
  xmin=0.85, xmax=38,
  xtick={{1,2,4,8,16,32}},
  xticklabels={{1,2,4,8,16,32}},
  x tick label style={{font=\tiny}},
  xlabel={{$\lambda/\eta^2$}},
  ylabel={{SubOpt ($10^{{-4}}$)}},
  ylabel style={{font=\tiny}},
  ymin=0.7, ymax=3.5
]
\addplot+[mairl active, mark=*, mark size=1.2pt,
  error bars/.cd, y dir=both, y explicit] coordinates {{{no_warm_coordinates}}};
\addplot[only marks, mark=o, mark size=2.7pt, line width=0.8pt,
         draw=mairlInk, forget plot] coordinates {{{no_warm_selected}}};
\end{{axis}}
\begin{{axis}}[
  at={{($(warmselect.south east)+(1.12cm,0)$)}},
  anchor=south west,
  mairl axis,
  width=0.34\textwidth,
  height=3.18cm,
  title={{(c) Held-out confirmation}},
  xmin=0.55, xmax=2.45,
  xtick={{1,2}},
  xticklabels={{Warm-up,No warm-up}},
  x tick label style={{font=\tiny}},
  ylabel={{Paired $\Delta$SubOpt ($10^{{-4}}$)}},
  ylabel style={{font=\tiny}},
  ymin=-2.1, ymax=1.5
]
\draw[mairlInk, densely dotted, line width=0.45pt]
  (axis cs:0.55,0) -- (axis cs:2.45,0);
\addplot+[mairl active, only marks, mark=*, mark size=1.7pt,
  error bars/.cd, y dir=both, y explicit] coordinates {{{effects}}};
\end{{axis}}
\end{{tikzpicture}}
"""


def render_zero_sum_spacious() -> str:
    """Use two source panels above a full-width downstream comparison."""

    path = REPOSITORY_ROOT / "reproducibility/data/zero_sum/comparison.json"
    payload = json.loads(path.read_text())
    source = payload["source"]
    downstream = payload["downstream"]["metrics"]
    fisher = _source_plots(source["fixed_leader_fisher_error"], "alt-fisher", legend=True)
    policy = _source_plots(source["regularized_qre_leader_l1_error"], "alt-policy", legend=False)
    deployment = _downstream_plots(
        downstream["stackelberg_suboptimality"],
        floor=1.0e-6,
        legend=False,
    )
    return rf"""% Exploratory alternative; generated from protected zero-sum data.
\begin{{minipage}}{{\textwidth}}
\centering
{{\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 1, group name=zssource, horizontal sep=1.45cm}},
  mairl log axis,
  width=0.445\textwidth,
  height=4.35cm
]
\nextgroupplot[
  title={{(a) Fixed-expert Fisher error}},
  xmin=0, xmax=200,
  ylabel={{Estimation error}},
  legend style={{at={{(0.98,0.98)}}, anchor=north east, font=\tiny, row sep=-1pt}}
]
{fisher}
\nextgroupplot[
  title={{(b) Source QRE policy error}},
  xmin=0, xmax=200,
  ylabel={{Leader-policy $\ell_1$ error}}
]
{policy}
\end{{groupplot}}
\node[font=\small, text=mairlInk, anchor=north]
  at ($(zssource c1r1.south)!0.5!(zssource c2r1.south)+(0,-0.38cm)$)
  {{Number of observed follower actions}};
\end{{tikzpicture}}}}

\vspace{{0.78cm}}

{{\begin{{tikzpicture}}
\begin{{axis}}[
  mairl axis,
  /pgfplots/ymode=log,
  width=0.92\textwidth,
  height=4.15cm,
  title={{(c) Transfer to downstream leader objectives}},
  xmin=0.55, xmax=3.45,
  xtick={{1,2,3}},
  xticklabels={{Task 1,Task 2,Task 3}},
  ylabel={{Final $\operatorname{{SubOpt}}(\hat\mu)$}},
  ymin=1.0e-6,
  legend style={{at={{(0.5,0.98)}}, anchor=north, legend columns=3,
                 font=\tiny, column sep=7pt}}
]
{deployment}
\end{{axis}}
\end{{tikzpicture}}}}
\end{{minipage}}
"""


def _whisker_series(
    values: np.ndarray,
    x: np.ndarray,
    *,
    offsets: tuple[float, ...],
    floor: float,
    legend: bool,
) -> str:
    commands: list[str] = []
    for method_index, ((_method, label, color), offset) in enumerate(
        zip(STRUCTURED_METHODS, offsets, strict=True)
    ):
        mean, lower, upper = mean_interval(values[:, method_index, :].T, 0.95)
        shifted = x + offset
        for x_value, low, high in zip(shifted, lower, upper, strict=True):
            commands.append(
                rf"\addplot[draw={color}, line width=0.4pt, forget plot] coordinates "
                rf"{{({x_value:.8g},{max(floor, low):.8g}) "
                rf"({x_value:.8g},{max(floor, high):.8g})}};"
            )
        commands.append(
            rf"\addplot[only marks, draw={color}, fill={color}, mark=*, "
            rf"mark size=1.25pt, line width=0.55pt] "
            rf"coordinates {{{coordinates(shifted, np.maximum(mean, floor))}}};"
        )
        if legend:
            commands.append(rf"\addlegendentry{{{label}}}")
    return "\n".join(commands)


def render_structured_whiskers() -> str:
    """Combine the two structured sensitivities with discrete CI whiskers."""

    warm_path = REPOSITORY_ROOT / "reproducibility/data/structured_toy/final_warmup.npz"
    eta_path = REPOSITORY_ROOT / "reproducibility/data/structured_toy/final_temperature.npz"
    with np.load(warm_path, allow_pickle=False) as warm_arrays:
        warm_values = warm_arrays["final__stackelberg_suboptimality"].copy()
    with np.load(eta_path, allow_pickle=False) as eta_arrays:
        eta_values = eta_arrays["final__stackelberg_suboptimality"].copy()
    warm = _whisker_series(
        warm_values,
        WARMUP_BUDGETS / 1000.0,
        offsets=(-0.007, 0.0, 0.007),
        floor=1.0e-5,
        legend=True,
    )
    temperature = _whisker_series(
        eta_values,
        ETA_VALUES,
        offsets=(-0.025, 0.0, 0.025),
        floor=0.0,
        legend=False,
    )
    return rf"""% Exploratory alternative; generated from protected structured-game data.
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 1, horizontal sep=1.50cm}},
  mairl axis,
  width=0.455\textwidth,
  height=5.05cm,
  ytick style={{draw=none}},
  minor y tick style={{draw=none}}
]
\nextgroupplot[
  title={{(a) Allocation to common warm-up}},
  xmin=-0.025, xmax=1.025,
  xtick={{0,0.25,0.5,0.75,1}},
  xlabel={{Warm-up fraction}},
  ymode=log,
  ymin=1e-5, ymax=3e-2,
  ylabel={{Final $\operatorname{{SubOpt}}(\hat\mu)$}},
  legend style={{at={{(0.98,0.98)}}, anchor=north east, font=\tiny, row sep=-1pt}}
]
\draw[mairlInk, dotted, line width=0.45pt]
  (axis cs:0.12,1e-5) -- (axis cs:0.12,3e-2);
\node[anchor=north west, font=\tiny, text=mairlMuted]
  at (axis cs:0.125,2.3e-2) {{canonical}};
{warm}
\nextgroupplot[
  title={{(b) Follower response sharpness}},
  xmin=0.43, xmax=3.07,
  xtick={{0.5,0.75,1.25,2,3}},
  xlabel={{Follower inverse temperature $\eta$}},
  ymin=0, ymax=0.0018,
  ylabel={{Final $\operatorname{{SubOpt}}(\hat\mu)$}}
]
\draw[mairlInk, dotted, line width=0.45pt]
  (axis cs:1.25,0) -- (axis cs:1.25,0.0018);
\node[anchor=north east, font=\tiny, text=mairlMuted]
  at (axis cs:1.23,0.00172) {{canonical}};
{temperature}
\end{{groupplot}}
\end{{tikzpicture}}
"""


def render_all() -> tuple[Path, ...]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "lobeke_relation.tex": render_lobeke_relation(),
        "lobeke_tuning_spacious.tex": render_tuning_spacious(),
        "structured_whiskers.tex": render_structured_whiskers(),
    }
    outputs = []
    for name, content in figures.items():
        path = OUTPUT_DIR / name
        path.write_text(content)
        outputs.append(path)
    return tuple(outputs)


def main() -> None:
    for path in render_all():
        print(path.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
