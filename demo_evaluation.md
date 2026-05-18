# Evaluation Demo

In the following, we use the benchmark instance labels provided by GBD's meta.db, and the solver runtimes in the 2023 SAT competition.



```python
from gbd_core.api import GBD
import polars as pl

def write_markdown(df: pl.DataFrame, path: str, floatfmt: str = ".2f"):
    columns = df.columns
    rows = df.rows()

    def fmt(value):
        return format(value, floatfmt) if isinstance(value, float) else str(value)

    rendered = [[fmt(value) for value in row] for row in rows]
    widths = [
        max(len(column), *(len(row[i]) for row in rendered)) if rendered else len(column)
        for i, column in enumerate(columns)
    ]
    aligns = [
        (":" + "-" * max(3, width + 1)) if i == 0 else ("-" * max(3, width + 1) + ":")
        for i, width in enumerate(widths)
    ]

    lines = [
        "| " + " | ".join(column.ljust(widths[i]) for i, column in enumerate(columns)) + " |",
        "|" + "|".join(aligns) + "|",
    ]
    for row in rendered:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) if i == 0 else row[i].rjust(widths[i]) for i in range(len(columns))) + " |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def get_solvers():
    with GBD(['data/sc2023/results_main_detailed.csv']) as gbd:
        return [ s for s in gbd.get_features() if s not in ["aresult", "vresult"] ]

def get_sc23_data(runtimes, metadata, addvbs=False):
    with GBD(['data/meta.db', 'data/sc2023/results_main_detailed.csv']) as gbd:
        df = gbd.query("track=main_2023", resolve=runtimes + metadata, collapse='min')
        # convert runtimes to numeric values
        df = df.with_columns(pl.col(runtimes).cast(pl.Float64, strict=False))
        # penalize timeouts
        df = df.with_columns([
            pl.when(pl.col(solver) >= 5000)
            .then(10000.0)
            .otherwise(pl.col(solver))
            .alias(solver)
            for solver in runtimes
        ])
        if addvbs:
            # compute the vbs score
            df = df.with_columns(pl.min_horizontal(*[pl.col(s) for s in runtimes]).alias("vbs"))
        return df
```

### Portfolio Analysis

Next, we determine the best 3-portfolio of solvers in the 2023 SAT competition.


```python
from itertools import combinations

def pscore(df: pl.DataFrame, solvers: list[str]):
    return df.select(pl.min_horizontal(*[pl.col(s) for s in solvers]).mean()).item()

def pscores_all(df: pl.DataFrame, solvers: list[str], k: int):
    return sorted([ (comb, pscore(df, list(comb))) for comb in combinations(solvers, k) ], key=lambda k : k[1])

def pscores_ext(df: pl.DataFrame, solvers: list[str], tuples: list[tuple[str]]):
    tupset = set(frozenset(comb + (s,)) for comb in tuples for s in solvers if s not in comb)
    return sorted([ (tuple(comb), pscore(df, list(comb))) for comb in tupset ], key=lambda k : k[1])

beam_width: int = 10
pf3 = get_solvers()
runtimes = get_sc23_data(pf3, [])
pfs1 = pscores_all(runtimes, pf3, 1)
pfs2 = pscores_all(runtimes, pf3, 2)[:beam_width]
pfs3 = pscores_ext(runtimes, pf3, [tup[0] for tup in pfs2])[:beam_width]

pf1, pf1score = list(pfs1[0][0]), round(pfs1[0][1], 2)
pf2, pf2score = list(pfs2[0][0]), round(pfs2[0][1], 2)
pf3, pf3score = list(pfs3[0][0]), round(pfs3[0][1], 2)

print("Single-best Solver and Best 2- and 3-Portfolios")
print(pf1score, pf1)
ordered = [ s[0][0] for s in pfs1 ]
print(pf2score, sorted(pf2, key=lambda s : ordered.index(s)))
print(pf3score, sorted(pf3, key=lambda s : ordered.index(s)))
```

    Single-best Solver and Best 2- and 3-Portfolios
    3274.01 ['SBVA_sbva_cadical']
    2434.72 ['SBVA_sbva_cadical', 'Kissat_MAB_prop_pr_no_sym']
    2138.96 ['SBVA_sbva_cadical', 'Kissat_MAB_prop_pr_no_sym', 'BreakID_kissat_low_sh']


### Category-wise Ranking

Determine the PAR-2 scores per instance category for each solver in the best 3-portfolio.


```python
df = get_sc23_data(pf3, ["family"])
# group families with less than 5 instances into a single group
misc = (
    df.group_by("family")
    .agg(pl.len().alias("count"))
    .filter(pl.col("count") < 5)
    .get_column("family")
    .to_list()
)
df = df.with_columns(
    pl.when(pl.col("family").is_in(misc))
    .then(pl.lit("miscellaneous"))
    .otherwise(pl.col("family"))
    .alias("family")
)
# compute family sizes and family-wise scores
counts = df.group_by("family").agg(pl.len().alias("count"))
groups = df.group_by("family").agg(pl.col(pf3).mean())
tab = groups.join(counts, on="family", how="left")
# sort families by the difference between the best and worst solver
tab = tab.with_columns((pl.max_horizontal(*[pl.col(s) for s in pf3]) - pl.min_horizontal(*[pl.col(s) for s in pf3])).alias("diff"))
tab = tab.sort("diff", descending=True)
tab = tab.select(["family", "count"] + pf3)
write_markdown(tab, "family_scores.md", floatfmt=".2f")
# output family_scores.md
with open("family_scores.md", "r") as f:
    print(f.read())
```

    | family                       | count | SBVA_sbva_cadical | Kissat_MAB_prop_pr_no_sym | BreakID_kissat_low_sh |
    |:-----------------------------|------:|------------------:|--------------------------:|----------------------:|
    | interval-matching            |    20 |          10000.00 |                      0.15 |              10000.00 |
    | or_randxor                   |     5 |             21.82 |                  10000.00 |                103.93 |
    | hashtable-safety             |    20 |            797.46 |                    194.75 |              10000.00 |
    | satcoin                      |    15 |           1395.53 |                  10000.00 |              10000.00 |
    | set-covering                 |    20 |            722.01 |                   5761.57 |                262.39 |
    | cryptography-ascon           |    20 |            356.82 |                   5628.35 |               2673.77 |
    | grs-fp-comm                  |    17 |           3649.90 |                   3435.82 |               8258.64 |
    | reg-n                        |     5 |          10000.00 |                   6295.16 |              10000.00 |
    | mutilated-chessboard         |    12 |           3194.54 |                   1656.50 |               5135.77 |
    | profitable-robust-production |    20 |           2470.42 |                   3946.63 |               5031.37 |
    | hardware-verification        |     8 |           2832.05 |                   1558.52 |               3964.51 |
    | register-allocation          |    20 |            101.20 |                      5.50 |               2016.39 |
    | tseitin                      |    11 |           8196.91 |                   8193.40 |               6393.20 |
    | social-golfer                |    20 |           7555.16 |                   7665.62 |               9013.05 |
    | miscellaneous                |    70 |           3164.75 |                   2563.98 |               3918.52 |
    | pigeon-hole                  |     8 |           5261.85 |                   6381.16 |               5128.03 |
    | school-timetabling           |    19 |           1399.65 |                   1266.09 |               2371.42 |
    | brent-equations              |    19 |            232.86 |                    408.33 |               1133.50 |
    | miter                        |    11 |           3134.91 |                   3157.68 |               3723.05 |
    | argumentation                |    20 |           3693.93 |                   4144.27 |               3820.58 |
    | subsumptiontest              |     5 |            230.22 |                     84.60 |                 89.14 |
    | planning                     |     6 |              6.97 |                     91.26 |                 10.98 |
    | quasigroup-completion        |     5 |              9.33 |                     60.78 |                  3.43 |
    | cryptography                 |     7 |           1578.46 |                   1577.96 |               1570.64 |
    | cryptography-simon           |    17 |          10000.00 |                  10000.00 |              10000.00 |