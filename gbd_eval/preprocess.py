# MIT License

# © 2023 Markus Iser, University of Helsinki

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

from gbd_core.api import GBD
import polars as pl

class DataPreprocessor:

    def __init__(self, gbd: GBD, query: str, features: list[str]):
        self.gbd = gbd
        self.query = query
        self.features = features
        self.df = gbd.query(query, resolve=features)

    def get(self):
        return self.df

    def numeric(self, columns: list[str]):
        self.df = self.df.with_columns(pl.col(columns).cast(pl.Float64, strict=False))
        return self
    
    def penalize(self, columns: list[str], max_runtime: int = 5000):
        self.df = self.df.with_columns([
            pl.when((pl.col(name) >= max_runtime) | (pl.col(name) < 0))
            .then(2 * max_runtime)
            .otherwise(pl.col(name))
            .alias(name)
            for name in columns
        ])
        return self

    def remainder(self, column: str, min_group_size: int = 5, rname: str = "miscellaneous"):
        small = (
            self.df.group_by(column)
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") < min_group_size)
            .get_column(column)
            .to_list()
        )
        small.extend(["empty", "unknown"])
        self.df = self.df.with_columns(
            pl.when(pl.col(column).is_in(small))
            .then(pl.lit(rname))
            .otherwise(pl.col(column))
            .alias(column)
        )
        return self
    
    def vbs(self, columns: list[str]):
        if set(columns) <= set(self.df.columns):
            self.df = self.df.with_columns(pl.min_horizontal(*[pl.col(name) for name in columns]).alias("vbs"))
        else:
            data = DataPreprocessor(self.gbd, self.query, columns)
            vbs = data.numeric(columns).penalize(columns).vbs(columns).get()
            self.df = self.df.with_columns(vbs.get_column("vbs"))
        return self
