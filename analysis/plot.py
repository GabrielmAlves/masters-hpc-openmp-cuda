#!/usr/bin/env python3
"""Gera os gráficos e o resumo dos experimentos a partir de results/raw/*.csv.

Uso: .venv/bin/python analysis/plot.py
Saídas: results/figures/*.png e results/summary.md

Todas as métricas usam a mediana das repetições de cada configuração.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
FIG = ROOT / "results" / "figures"

# Paleta categórica validada (ordem fixa) e rampa sequencial azul.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "font.size": 10,
})


# ---------------------------------------------------------------- dados

def load(name):
    path = RAW / f"{name}.csv"
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("N", "iters", "threads", "bx", "by", "rep"):
            r[k] = int(r[k])
        for k in ("t_loop", "t_transfer", "mlups"):
            r[k] = float(r[k])
    return rows


def median_by(rows, keys, field="t_loop"):
    """Mediana de `field` agrupada pelas chaves."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r[field])
    return {k: statistics.median(v) for k, v in groups.items()}


def mlups(n, iters, t):
    return (n - 2) ** 2 * iters / t / 1e6


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


def size_label(n):
    mb = 2 * n * n * 8 / 2**20
    return f"{n}\n({mb:.0f} MB)" if mb < 1024 else f"{n}\n({mb / 1024:.0f} GB)"


seq = load("seq")
omp = load("omp_threads")
aff = load("omp_affinity")
otemp = load("omp_temporal")
vec = load("vectorization")
cblk = load("cuda_blocks")
ccoar = load("cuda_coarsen")
csize = load("cuda_size")

seq_t = {k[0]: v for k, v in median_by(seq, ["N"]).items()}
iters_of = {r["N"]: r["iters"] for r in seq}
sizes = sorted(seq_t)
summary = []


def table(header, rows):
    summary.append("| " + " | ".join(header) + " |")
    summary.append("|" + "---|" * len(header))
    for row in rows:
        summary.append("| " + " | ".join(str(c) for c in row) + " |")
    summary.append("")


def fmt(x, d=2):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------- OpenMP

omp_t = median_by(omp, ["variant", "N", "threads"])
omp_variants = [v for v in ("basic", "single") if any(k[0] == v for k in omp_t)]
omp_variants += sorted({k[0] for k in omp_t if k[0].startswith("temporal")})
threads = sorted({k[2] for k in omp_t})
# Tamanhos com todas as variantes medidas (o experimento pode estar incompleto).
o_sizes = [n for n in sizes if all((v, n, t) in omp_t for v in omp_variants for t in threads)]

if omp_t:
    print("OpenMP")
    # Speedup em relação à sequencial: um painel por tamanho.
    cols = 3
    rows_n = (len(o_sizes) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(11, 3.3 * rows_n), sharex=True)
    axes = axes.flatten()
    for ax, n in zip(axes, o_sizes):
        for i, v in enumerate(omp_variants):
            ys = [seq_t[n] / omp_t[(v, n, t)] for t in threads if (v, n, t) in omp_t]
            ax.plot(threads[:len(ys)], ys, marker="o", color=SERIES[i], label=v)
        ax.plot(threads, threads, color=INK2, linewidth=1, linestyle=":", label="ideal")
        ax.set_title(f"N = {n} ({2 * n * n * 8 / 2**20:.0f} MB)")
        ax.set_ylim(bottom=0)
        ax.set_xticks([1, 4, 8, 12, 16])
    for ax in axes[len(o_sizes):]:
        ax.set_visible(False)
    for ax in axes[-cols:]:
        ax.set_xlabel("threads")
    for ax in axes[::cols]:
        ax.set_ylabel("speedup sobre a sequencial")
    axes[0].legend(loc="upper left")
    fig.suptitle("OpenMP: speedup em função do número de threads", y=1.0)
    fig.tight_layout()
    save(fig, "omp_speedup_threads")

    # Eficiência da variante basic, uma linha por tamanho (rampa sequencial).
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, v in zip(axes, ["basic", omp_variants[-1]]):
        for i, n in enumerate(o_sizes):
            ys = [seq_t[n] / omp_t[(v, n, t)] / t for t in threads if (v, n, t) in omp_t]
            ax.plot(threads[:len(ys)], ys, marker="o", color=SEQ[i % len(SEQ)], label=f"N={n}")
        ax.set_title(f"Eficiência: {v}")
        ax.set_xlabel("threads")
        ax.set_xticks([1, 4, 8, 12, 16])
    axes[0].set_ylabel("eficiência (speedup / threads)")
    axes[1].legend(ncol=2)
    fig.tight_layout()
    save(fig, "omp_efficiency")

    # Resumo: melhor configuração OpenMP por tamanho.
    summary.append("## OpenMP: melhor configuração por tamanho\n")
    rows_tab = []
    for n in o_sizes:
        cands = {(v, t): omp_t[(v, n, t)] for v in omp_variants for t in threads if (v, n, t) in omp_t}
        (bv, bt), best = min(cands.items(), key=lambda kv: kv[1])
        basic_best_t = min((t for t in threads if ("basic", n, t) in omp_t), key=lambda t: omp_t[("basic", n, t)])
        b = omp_t[("basic", n, basic_best_t)]
        rows_tab.append([n, fmt(seq_t[n], 3), f"{fmt(b, 3)} ({basic_best_t} t)", fmt(seq_t[n] / b),
                         f"{bv}, {bt} t", fmt(best, 3), fmt(seq_t[n] / best), fmt(mlups(n, iters_of[n], best), 0)])
    table(["N", "seq (s)", "basic: melhor (s)", "speedup basic", "melhor variante", "tempo (s)",
           "speedup", "MLUPS"], rows_tab)

    summary.append("## OpenMP: speedup por número de threads (variante basic)\n")
    table(["N"] + [f"{t} t" for t in threads],
          [[n] + [fmt(seq_t[n] / omp_t[("basic", n, t)]) if ("basic", n, t) in omp_t else "-" for t in threads]
           for n in o_sizes])
    if omp_variants[-1] != "basic":
        v = omp_variants[-1]
        summary.append(f"## OpenMP: speedup por número de threads ({v})\n")
        table(["N"] + [f"{t} t" for t in threads],
              [[n] + [fmt(seq_t[n] / omp_t[(v, n, t)]) if (v, n, t) in omp_t else "-" for t in threads]
               for n in o_sizes])

# Afinidade
if aff:
    print("Afinidade")
    aff_t = median_by(aff, ["setting", "variant", "N", "threads"])
    settings = ["none", "close:cores", "spread:cores", "close:threads"]
    a_sizes = sorted({k[2] for k in aff_t})
    a_vars = sorted({k[1] for k in aff_t}, key=lambda v: (not v.startswith("basic"), v))
    fig, axes = plt.subplots(len(a_vars), len(a_sizes), figsize=(11, 3.4 * len(a_vars)), squeeze=False)
    for r, v in enumerate(a_vars):
        for c, n in enumerate(a_sizes):
            ax = axes[r][c]
            for i, s in enumerate(settings):
                ts = sorted(t for (ss, vv, nn, t) in aff_t if ss == s and vv == v and nn == n)
                ax.plot(ts, [mlups(n, iters_of.get(n, 1), aff_t[(s, v, n, t)]) for t in ts],
                        marker="o", color=SERIES[i], label=s)
            ax.set_title(f"{v}, N = {n}")
            ax.set_xlabel("threads")
            ax.set_ylabel("MLUPS")
            ax.set_ylim(bottom=0)
            ax.set_xticks([1, 4, 8, 12, 16])
    axes[0][0].legend(title="OMP_PROC_BIND:OMP_PLACES", title_fontsize=8)
    fig.tight_layout()
    save(fig, "omp_affinity")

    summary.append("## OpenMP: afinidade (MLUPS, mediana)\n")
    rows_tab = []
    for v in a_vars:
        for n in a_sizes:
            for t in sorted({k[3] for k in aff_t}):
                rows_tab.append([v, n, t] + [fmt(mlups(n, iters_of.get(n, 1), aff_t[(s, v, n, t)]), 0)
                                             if (s, v, n, t) in aff_t else "-" for s in settings])
    table(["variante", "N", "threads"] + settings, rows_tab)

# Blocagem temporal: mapa de calor tile x tsteps
if otemp:
    print("Blocagem temporal OpenMP")
    ot = defaultdict(list)
    for r in otemp:
        _, b, t = r["variant"].split("_")
        ot[(r["N"], int(b[1:]), int(t[1:]))].append(r["mlups"])
    ot = {k: statistics.median(v) for k, v in ot.items()}
    t_sizes = sorted({k[0] for k in ot})
    tiles = sorted({k[1] for k in ot})
    steps = sorted({k[2] for k in ot})
    fig, axes = plt.subplots(1, len(t_sizes), figsize=(5.4 * len(t_sizes), 3.6))
    axes = axes if len(t_sizes) > 1 else [axes]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#e8f1fc"] + SEQ)
    for ax, n in zip(axes, t_sizes):
        grid = [[ot.get((n, b, s), float("nan")) / 1000 for s in steps] for b in tiles]
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower")
        vmax = max(max(row) for row in grid)
        for i, b in enumerate(tiles):
            for j, s in enumerate(steps):
                val = grid[i][j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if val > 0.55 * vmax else INK)
        ax.set_xticks(range(len(steps)), steps)
        ax.set_yticks(range(len(tiles)), tiles)
        ax.set_xlabel("tsteps (iterações por bloco)")
        ax.set_ylabel("tile (lado do bloco)")
        ax.set_title(f"N = {n}: GLUPS, 8 threads")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="GLUPS")
    fig.tight_layout()
    save(fig, "omp_temporal_params")

    summary.append("## OpenMP: blocagem temporal, melhor tile x tsteps (8 threads)\n")
    rows_tab = []
    for n in t_sizes:
        (b, s), best = max((((k[1], k[2]), v) for k, v in ot.items() if k[0] == n), key=lambda kv: kv[1])
        rows_tab.append([n, b, s, fmt(best, 0), fmt(ot.get((n, b, 1), float("nan")), 0)])
    table(["N", "tile", "tsteps", "MLUPS", "MLUPS com tsteps=1 (sem reuso temporal)"], rows_tab)

# Vetorização
if vec:
    print("Vetorização")
    vt = median_by(vec, ["setting", "version", "N"])
    builds = ["novec", "sse2", "avx2"]
    labels = {"novec": "sem vetorização", "sse2": "SSE2 (128 bits)", "avx2": "AVX2 (256 bits)"}
    v_sizes = sorted({k[2] for k in vt})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    width = 0.26
    for ax, version, title in zip(axes, ["seq", "omp"], ["Sequencial", "OpenMP basic, 8 threads"]):
        for i, b in enumerate(builds):
            xs = [k + (i - 1) * (width + 0.02) for k in range(len(v_sizes))]
            ys = [mlups(n, iters_of.get(n, 1), vt[(b, version, n)]) if (b, version, n) in vt else 0 for n in v_sizes]
            ax.bar(xs, ys, width=width, color=SERIES[i], label=labels[b])
        ax.set_xticks(range(len(v_sizes)), [size_label(n) for n in v_sizes], fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("MLUPS")
        ax.grid(axis="x", visible=False)
    axes[0].legend()
    fig.tight_layout()
    save(fig, "vectorization")

    summary.append("## Vetorização (MLUPS, mediana)\n")
    rows_tab = []
    for version in ["seq", "omp"]:
        for n in v_sizes:
            vals = [mlups(n, iters_of.get(n, 1), vt[(b, version, n)]) for b in builds]
            rows_tab.append([version, n] + [fmt(x, 0) for x in vals] + [fmt(vals[2] / vals[0])])
    table(["versão", "N", "sem vetorização", "SSE2", "AVX2", "AVX2 / sem vet."], rows_tab)

# ---------------------------------------------------------------- CUDA

if cblk:
    print("CUDA: blocos")
    bt = median_by(cblk, ["variant", "N", "bx", "by"], "mlups")
    kernels = ["basic", "int32", "shared", "coarse_c4"]
    b_sizes = sorted({k[1] for k in bt})
    blocks = sorted({(k[2], k[3]) for k in bt if k[0] in kernels}, key=lambda b: (b[0] * b[1], b[0]))
    fig, axes = plt.subplots(len(b_sizes), 1, figsize=(11, 3.4 * len(b_sizes)), squeeze=False)
    width = 0.2
    for ax, n in zip(axes[:, 0], b_sizes):
        for i, k in enumerate(kernels):
            xs = [j + (i - 1.5) * (width + 0.01) for j in range(len(blocks))]
            ys = [bt.get((k, n, bx, by), 0) / 1000 for bx, by in blocks]
            ax.bar(xs, ys, width=width, color=SERIES[i], label=k)
        ax.set_xticks(range(len(blocks)), [f"{bx}×{by}" for bx, by in blocks], fontsize=8)
        ax.set_ylabel("GLUPS")
        ax.set_title(f"CUDA, double, N = {n}: dimensões do bloco")
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, "cuda_blocks")

    temporal = {k: v for k, v in bt.items() if k[0].startswith("temporal")}
    if temporal:
        fig, axes = plt.subplots(1, len(b_sizes), figsize=(5.5 * len(b_sizes), 3.6), squeeze=False)
        for ax, n in zip(axes[0], b_sizes):
            tblocks = sorted({(k[2], k[3]) for k in temporal if k[1] == n}, key=lambda b: (b[0] * b[1], b[0]))
            tsteps_all = sorted({int(k[0][10:]) for k in temporal})
            for i, s in enumerate(tsteps_all):
                pts = [(j, temporal[(f"temporal_t{s}", n, bx, by)] / 1000) for j, (bx, by) in enumerate(tblocks)
                       if (f"temporal_t{s}", n, bx, by) in temporal]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=SEQ[min(i * 2, 5)],
                        label=f"tsteps={s}")
            ref = max(v for k, v in bt.items() if k[0] == "int32" and k[1] == n) / 1000
            ax.axhline(ref, color=SERIES[1], linewidth=1.5, linestyle="--", label="int32 (melhor bloco)")
            ax.set_xticks(range(len(tblocks)), [f"{bx}×{by}" for bx, by in tblocks], fontsize=8)
            ax.set_ylabel("GLUPS")
            ax.set_ylim(bottom=0)
            ax.set_title(f"CUDA temporal, N = {n}")
        axes[0][0].legend(fontsize=8)
        fig.tight_layout()
        save(fig, "cuda_temporal")

    summary.append("## CUDA: melhor bloco por kernel (double, GLUPS)\n")
    rows_tab = []
    for n in b_sizes:
        for k in sorted({kk[0] for kk in bt}):
            cands = {(kk[2], kk[3]): v for kk, v in bt.items() if kk[0] == k and kk[1] == n}
            if not cands:
                continue
            (bx, by), best = max(cands.items(), key=lambda kv: kv[1])
            worst = min(cands.values())
            rows_tab.append([n, k, f"{bx}×{by}", fmt(best / 1000), fmt(worst / 1000)])
    table(["N", "kernel", "melhor bloco", "GLUPS (melhor)", "GLUPS (pior bloco)"], rows_tab)

if ccoar:
    print("CUDA: coarsening")
    ct = median_by(ccoar, ["variant", "N", "bx", "by"], "mlups")
    c_sizes = sorted({k[1] for k in ct})
    fig, axes = plt.subplots(1, len(c_sizes), figsize=(5.5 * len(c_sizes), 3.4), squeeze=False)
    for ax, n in zip(axes[0], c_sizes):
        cblocks = sorted({(k[2], k[3]) for k in ct if k[1] == n})
        for i, (bx, by) in enumerate(cblocks):
            cs = sorted({int(k[0].split("_c")[1]) for k in ct if k[1] == n and (k[2], k[3]) == (bx, by)})
            ax.plot(cs, [ct[(f"coarse_c{c}", n, bx, by)] / 1000 for c in cs], marker="o",
                    color=SERIES[i], label=f"{bx}×{by}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_xlabel("linhas por thread (coarsen)")
        ax.set_ylabel("GLUPS")
        ax.set_ylim(bottom=0)
        ax.set_title(f"CUDA coarse, N = {n}")
    axes[0][0].legend(title="bloco")
    fig.tight_layout()
    save(fig, "cuda_coarsen")

if csize:
    print("CUDA: tamanhos")
    st = median_by(csize, ["variant", "prec", "N"])
    stt = median_by(csize, ["variant", "prec", "N"], "t_transfer")
    s_sizes = sorted({k[2] for k in st})

    def series_of(prefix, prec):
        keys = sorted({k[0] for k in st if k[1] == prec and k[0].split("+")[0].startswith(prefix)
                       and "+" not in k[0]})
        return keys[0] if keys else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    kern = [k for k in (series_of(p, "double") for p in
                        ("basic", "int32", "shared", "coarse", "temporal", "uncoalesced")) if k]
    for i, k in enumerate(kern):
        ax = axes[0]
        ns = [n for n in s_sizes if (k, "double", n) in st]
        ax.plot(ns, [mlups(n, iters_of[n], st[(k, "double", n)]) / 1000 for n in ns], marker="o",
                color=SERIES[i], label=k)
    axes[0].set_title("Kernels em double")
    extra = [(series_of("int32", "double"), "double"), (series_of("int32", "float"), "float"),
             ("int32+graph", "double"), ("int32+graph+pinned", "float")]
    names = ["int32 double", "int32 float", "int32 double + graph", "int32 float + graph"]
    for i, ((k, p), name) in enumerate(zip(extra, names)):
        ns = [n for n in s_sizes if (k, p, n) in st]
        axes[1].plot(ns, [mlups(n, iters_of[n], st[(k, p, n)]) / 1000 for n in ns], marker="o",
                     color=SERIES[i], label=name)
    axes[1].set_title("Precisão e CUDA Graphs")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(s_sizes, [str(n) for n in s_sizes])
        ax.set_xlabel("N")
        ax.set_ylabel("GLUPS (só o laço)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "cuda_size")

    # Transferências: paginável x pinned, e fração do tempo total.
    k32 = series_of("int32", "double")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    width = 0.38
    for i, (k, name) in enumerate([(k32, "paginável"), ("int32+pinned", "pinned")]):
        xs = [j + (i - 0.5) * (width + 0.02) for j in range(len(s_sizes))]
        loop = [st[(k, "double", n)] * 1000 for n in s_sizes]
        trans = [stt[(k, "double", n)] * 1000 for n in s_sizes]
        axes[0].bar(xs, trans, width=width, color=SERIES[i], label=name)
        frac = [t / (t + lp) * 100 for t, lp in zip(trans, loop)]
        axes[1].bar(xs, frac, width=width, color=SERIES[i], label=name)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("tempo de transferência (ms)")
    axes[0].set_title("Transferências host ↔ GPU (3 cópias)")
    axes[1].set_ylabel("% do tempo total")
    axes[1].set_title("Fração do tempo gasta em transferências")
    for ax in axes:
        ax.set_xticks(range(len(s_sizes)), [size_label(n) for n in s_sizes], fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.legend()
    fig.tight_layout()
    save(fig, "cuda_transfer")

    summary.append("## CUDA por tamanho (GLUPS, só o laço; mediana)\n")
    cols_k = sorted({(k[0], k[1]) for k in st}, key=lambda kp: (kp[1], kp[0]))
    table(["N"] + [f"{k} ({p})" for k, p in cols_k],
          [[n] + [fmt(mlups(n, iters_of[n], st[(k, p, n)]) / 1000) if (k, p, n) in st else "-" for k, p in cols_k]
           for n in s_sizes])

    summary.append("## CUDA: transferências (ms, 2 envios + 1 retorno)\n")
    table(["N", "MB por cópia", "paginável", "pinned", "laço (int32)", "transf. / total (paginável)"],
          [[n, fmt(n * n * 8 / 2**20, 0), fmt(stt[(k32, "double", n)] * 1000), fmt(stt[("int32+pinned", "double", n)] * 1000),
            fmt(st[(k32, "double", n)] * 1000),
            f"{fmt(stt[(k32, 'double', n)] / (stt[(k32, 'double', n)] + st[(k32, 'double', n)]) * 100, 1)} %"]
           for n in s_sizes])

# ---------------------------------------------------------------- comparação geral

if seq_t and omp_t and csize:
    print("Comparação geral")
    best_omp = {n: min(v for k, v in omp_t.items() if k[1] == n) for n in o_sizes}
    basic16 = {n: omp_t.get(("basic", n, max(threads)), float("nan")) for n in o_sizes}
    k32 = series_of("int32", "double")
    entries = [
        ("Sequencial", seq_t),
        (f"OpenMP basic ({max(threads)} threads)", basic16),
        ("OpenMP: melhor variante", best_omp),
        ("CUDA basic (inicial)", {n: st[("basic", "double", n)] for n in s_sizes}),
        ("CUDA: melhor double", {n: min(v for k, v in st.items() if k[1] == "double" and k[2] == n) for n in s_sizes}),
        ("CUDA: melhor float", {n: min(v for k, v in st.items() if k[1] == "float" and k[2] == n) for n in s_sizes}),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (name, d) in enumerate(entries):
        ns = [n for n in sizes if n in d]
        axes[0].plot(ns, [mlups(n, iters_of[n], d[n]) / 1000 for n in ns], marker="o", color=SERIES[i], label=name)
        axes[1].plot(ns, [seq_t[n] / d[n] for n in ns], marker="o", color=SERIES[i], label=name)
    axes[0].set_ylabel("GLUPS (só o laço)")
    axes[0].set_title("Desempenho por tamanho")
    axes[1].set_ylabel("speedup sobre a sequencial")
    axes[1].set_yscale("log")
    axes[1].set_title("Speedup por tamanho")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.set_xlabel("N")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "overview")

    summary.insert(0, "")
    rows_tab = []
    for n in sizes:
        row = [n, fmt(seq_t[n], 3)]
        for name, d in entries[1:]:
            row.append(f"{fmt(d[n], 4)} ({fmt(seq_t[n] / d[n], 1)}×)" if n in d else "-")
        rows_tab.append(row)
    header = ["N", "seq (s)"] + [e[0] for e in entries[1:]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows_tab]
    summary[0:0] = ["## Comparação geral: tempo do laço (s) e speedup sobre a sequencial\n"] + lines + [""]

# ---------------------------------------------------------------- validação

val_lines = []
for name in ("validation", "raw/validation_runs"):
    path = ROOT / "results" / f"{name}.csv"
    if not path.exists():
        continue
    with path.open() as f:
        vrows = list(csv.DictReader(f))
    ok = sum(r.get("resultado") == "OK" for r in vrows)
    val_lines.append(f"- `results/{name}.csv`: {ok} de {len(vrows)} casos OK.")
    if name == "validation":
        dbl = [r for r in vrows if r["prec"] == "double"]
        ident = sum(r["identico"] == "sim" for r in dbl)
        flt = [r for r in vrows if r["prec"] == "float"]
        val_lines.append(f"  - double: {ident} de {len(dbl)} idênticos bit a bit à sequencial "
                         f"(maior erro absoluto {max(float(r['erro_abs']) for r in dbl):.3e}).")
        if flt:
            val_lines.append(f"  - float: maior erro absoluto {max(float(r['erro_abs']) for r in flt):.3e}, "
                             f"maior erro relativo {max(float(r['erro_rel']) for r in flt):.3e}.")
    else:
        for prec, cond in (("double", lambda r: r["tol"] == "1e-12"), ("float", lambda r: r["tol"] != "1e-12")):
            sel = [r for r in vrows if cond(r)]
            if sel:
                val_lines.append(f"  - {prec}: maior erro absoluto {max(float(r['erro_abs']) for r in sel):.3e}, "
                                 f"maior erro relativo {max(float(r['erro_rel']) for r in sel):.3e} "
                                 f"(N até {max(int(r['N']) for r in sel)}).")

head = ["# Resumo dos experimentos", "",
        "Gerado por `analysis/plot.py` a partir de `results/raw/`. Tempos: mediana das repetições. "
        "MLUPS = (N-2)² × iterações / tempo do laço / 10⁶. Gráficos em `results/figures/`.", "",
        "## Validação", ""] + val_lines + [""]
(ROOT / "results" / "summary.md").write_text("\n".join(head + summary) + "\n", encoding="utf-8")
print("results/summary.md")
