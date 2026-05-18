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

from gbd_eval.util import name


def _format_value(value, precision=2):
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return name(value) if isinstance(value, str) else str(value)


def _latex_rows(df: pl.DataFrame, precision=2, bold_min_of: list[str] = None):
    rows = []
    for row in df.iter_rows(named=True):
        min_value = None
        if bold_min_of is not None:
            values = [row[col] for col in bold_min_of if col in row and isinstance(row[col], (int, float))]
            min_value = min(values) if values else None
        values = []
        for col in df.columns:
            value = _format_value(row[col], precision)
            if min_value is not None and col in bold_min_of and row[col] == min_value:
                value = "\\textbf{" + value + "}"
            values.append(value)
        rows.append(" & ".join(values) + r" \\")
    return rows


def _write_latex(df: pl.DataFrame, path: str, column_format: str, precision=2, bold_min_of: list[str] = None):
    headers = " & ".join(name(col) for col in df.columns) + r" \\"
    with open(path, "w") as f:
        f.write("\\begin{tabular}{" + column_format + "}\n")
        f.write("\\toprule\n")
        f.write(headers + "\n")
        f.write("\\midrule\n")
        f.write("\n".join(_latex_rows(df, precision, bold_min_of)) + "\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def _write_html(df: pl.DataFrame, path: str, precision=2):
    with open(path, "w") as f:
        f.write("<table>\n<thead><tr>")
        f.write("".join(f"<th>{name(col)}</th>" for col in df.columns))
        f.write("</tr></thead>\n<tbody>\n")
        for row in df.iter_rows(named=True):
            f.write("<tr>" + "".join(f"<td>{_format_value(row[col], precision)}</td>" for col in df.columns) + "</tr>\n")
        f.write("</tbody>\n</table>\n")


def scores(df: pl.DataFrame, to_latex: str = None, to_html: str = None):
    if to_html is not None:
        _write_html(df, to_html)
    if to_latex is not None:
        _write_latex(df, to_latex, column_format="l|" + "r" * (df.width - 1))


def group_wise_scores(df: pl.DataFrame, solvers: list[str], groups: list[str], to_latex: str, bold_min_of: list[str] = None, min_diff=0):
    df = df.with_columns(
        (pl.max_horizontal(*[pl.col(solver) for solver in solvers]) - pl.min_horizontal(*[pl.col(solver) for solver in solvers])).alias("diff")
    )
    df = df.filter(pl.col("diff") >= min_diff).drop("diff")

    width = "{:.2f}".format(.6 / len(solvers))
    sformat = ">{\\raggedleft\\arraybackslash}p{" + width + "\\linewidth}"
    column_format = "l" * len(groups) + "r|" + sformat * (len(solvers)-1) + "|r"
    _write_latex(df, to_latex, column_format=column_format, bold_min_of=bold_min_of)


def best_k_portfolios(df: pl.DataFrame, to_latex: str):
    _write_latex(df, to_latex, column_format="l|p{.9\\linewidth}|r")
    return df
