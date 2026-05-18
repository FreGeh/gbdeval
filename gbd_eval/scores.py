# MIT License
#
# © 2023 Markus Iser, University of Helsinki
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# run: ./eval.py

import polars as pl


def scores(df: pl.DataFrame):
    numeric_cols = [
        name for name, dtype in df.schema.items()
        if dtype.is_numeric()
    ]
    return pl.DataFrame({
        "solver": numeric_cols,
        "score": [df.get_column(name).mean() for name in numeric_cols],
    })

    
def scores_group_wise(df: pl.DataFrame, solvers: list[str], groups: list[str], sortby: str = "count"):
    group = groups[0]
    score_cols = solvers + (["vbs"] if "vbs" in df.columns else [])
    counts = df.group_by(group).agg(pl.len().alias("count"))
    means = df.group_by(group).agg(pl.col(score_cols).mean())
    tab = means.join(counts, on=group, how="left")
    tab = tab.with_columns([
        (pl.max_horizontal(*[pl.col(solver) for solver in solvers]) - pl.min_horizontal(*[pl.col(solver) for solver in solvers])).alias("diff"),
        (pl.max_horizontal(*[pl.col(solver) for solver in solvers]) / pl.min_horizontal(*[pl.col(solver) for solver in solvers])).alias("quot"),
        (pl.concat_list([pl.col(solver) for solver in solvers]).list.median() - pl.min_horizontal(*[pl.col(solver) for solver in solvers])).alias("diff2"),
        (pl.concat_list([pl.col(solver) for solver in solvers]).list.median() / pl.min_horizontal(*[pl.col(solver) for solver in solvers])).alias("quot2"),
    ])
    tab = tab.sort(sortby, descending=True).select([group, "count"] + score_cols)
    all_row = {group: "all", "count": df.height}
    all_row.update({solver: df.get_column(solver).mean() for solver in score_cols})
    return pl.concat([tab, pl.DataFrame([all_row]).select(tab.columns)], how="vertical_relaxed")
