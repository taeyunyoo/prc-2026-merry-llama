"""Metadata-only imputation ported from v21 notebook cells 21 and 22.

Both donor passes can be fitted before either is applied: pass 1 changes only
AOBT-missing rows, which can never become pass-2 donors.
No target, date, airport, phase, or fuzzy similarity is used in the lookup.
"""
from __future__ import annotations

import json
import platform

import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from .data import ProjectPaths, dataset_names, metadata_donor_names

KEY = "FLIGHT_mvt"
AOBT = "AOBT_3_flt"
FEATURES = [
    "CALLSIGN_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
]
REQUIRED = [KEY, AOBT, *FEATURES]
MISSING_TEXT = {"", "NAN", "NONE", "NULL", "<NA>", "NAT"}


def missing_mask(series):
    text = series.astype("string").str.strip().str.upper()
    return series.isna() | text.isin(MISSING_TEXT)


def flight_key(series):
    return series.astype("string").str.strip().mask(missing_mask(series))


def column_map(columns):
    lookup = {}
    for column in columns:
        lowered = column.lower()
        if lowered in lookup:
            raise ValueError(f"Ambiguous column names: {column}, {lookup[lowered]}")
        lookup[lowered] = column
    absent = [c for c in REQUIRED if c.lower() not in lookup]
    if absent:
        raise ValueError(f"Required columns missing: {absent}")
    return {c: lookup[c.lower()] for c in REQUIRED}


def donor_rows(df, mapping):
    d = df[list(mapping.values())].rename(
        columns={actual: logical for logical, actual in mapping.items()}
    )
    keys = flight_key(d[KEY])
    eligible = keys.notna() & ~missing_mask(d[AOBT])
    part = d.loc[eligible, FEATURES].copy()
    part.insert(0, KEY, keys.loc[eligible])
    for feature in FEATURES:
        part[feature] = part[feature].astype("string").mask(missing_mask(part[feature]))
    return part


def fit_lookups(donors):
    if donors.empty:
        raise ValueError("No AOBT-present donors with a flight key.")
    complete = donors.loc[donors[FEATURES].notna().all(axis=1)]
    if complete.empty:
        raise ValueError("No complete donors for the original v21 first pass.")
    grouped_complete = complete.groupby(KEY, sort=False)
    safe = grouped_complete[FEATURES].nunique(dropna=False).eq(1).all(axis=1)
    first = grouped_complete[FEATURES].first().loc[safe]

    grouped = donors.groupby(KEY, sort=False)
    counts = grouped.size()
    second = {}
    for feature in FEATURES:
        valid = grouped[feature].count().eq(counts) & grouped[feature].nunique().eq(1)
        second[feature] = grouped[feature].first().loc[valid]
    return first, second


def apply_lookups(df, mapping, first, second):
    """Mutate only missing metadata cells on AOBT-missing rows; preserve order."""
    keys = flight_key(df[mapping[KEY]])
    target = missing_mask(df[mapping[AOBT]])
    changed = pd.Series(False, index=df.index)
    summary = {"rows": len(df), "aobt_missing_rows": int(target.sum()),
               "missing_key_rows": int((target & keys.isna()).sum())}
    for label, lookups in (("pass1", first), ("pass2", second)):
        pass_changed = pd.Series(False, index=df.index)
        fills = {}
        for feature in FEATURES:
            column = mapping[feature]
            replacements = keys.map(lookups[feature])
            fill = target & missing_mask(df[column]) & replacements.notna()
            count = int(fill.sum())
            if count:
                if isinstance(df[column].dtype, pd.CategoricalDtype):
                    extra = [v for v in replacements.loc[fill].unique()
                             if v not in df[column].cat.categories]
                    if extra:
                        df[column] = df[column].cat.add_categories(extra)
                elif not isinstance(df[column].dtype, pd.StringDtype):
                    df[column] = df[column].astype(object)
                df.loc[fill, column] = replacements.loc[fill]
                pass_changed |= fill
            fills[feature] = count
        changed |= pass_changed
        summary[label] = {"changed_rows": int(pass_changed.sum()),
                          "filled_cells": sum(fills.values()), "by_feature": fills}
    summary["changed_rows"] = int(changed.sum())
    summary["filled_cells"] = sum(summary[p]["filled_cells"] for p in ("pass1", "pass2"))
    summary["remaining_by_feature"] = {
        f: int((target & missing_mask(df[mapping[f]])).sum()) for f in FEATURES
    }
    return df, summary


def run_preprocessing(root=None, overwrite=False, flight_set="all"):
    paths = ProjectPaths.from_root(root)
    source = paths.raw_dir
    destination = paths.metadata_filled_dir
    donor_names = metadata_donor_names()
    target_names = dataset_names(flight_set)
    required_names = list(dict.fromkeys(donor_names + target_names))
    missing = [str(source / name) for name in required_names if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing raw files:\n" + "\n".join(missing))
    mappings = {}
    for name in required_names:
        try:
            mappings[name] = column_map(pq.read_schema(source / name).names)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
    selected = "all" if flight_set is None else getattr(flight_set, "value", flight_set)
    summary_name = "preprocessing_summary.json" if selected == "all" else f"preprocessing_summary_{selected}.json"
    summary_path = destination / summary_name
    existing = [destination / n for n in target_names if (destination / n).exists()]
    if (existing or summary_path.exists()) and not overwrite:
        raise FileExistsError("Outputs already exist. Use --overwrite to rebuild from raw.")

    parts = []
    donor_counts = {}
    print("Donors: 2025 training + ranking; final_ranking is target-only.", flush=True)
    for number, name in enumerate(donor_names, 1):
        mapping = mappings[name]
        df = pd.read_parquet(source / name, columns=list(mapping.values()))
        part = donor_rows(df, mapping)
        donor_counts[name] = len(part)
        # Repeated identical metadata cannot affect either unanimity rule.
        parts.append(part.drop_duplicates())
        print(f"DONOR {number}/13 {name}: {len(part):,}", flush=True)
        del df, part
    donors = pd.concat(parts, ignore_index=True).drop_duplicates()
    del parts
    first, second = fit_lookups(donors)
    del donors
    print(f"LOOKUP pass1 keys: {len(first):,}; pass2: "
          + str({f: len(v) for f, v in second.items()}), flush=True)

    destination.mkdir(parents=True, exist_ok=True)
    # Any partial rebuild also invalidates the previous all-datasets marker.
    summary_path.unlink(missing_ok=True)
    (destination / "preprocessing_summary.json").unlink(missing_ok=True)
    if selected == "all":
        for group in ("train", "rank", "final"):
            (destination / f"preprocessing_summary_{group}.json").unlink(missing_ok=True)
    records = []
    for number, name in enumerate(target_names, 1):
        df = pd.read_parquet(source / name)
        df, stats = apply_lookups(df, mappings[name], first, second)
        output = destination / name
        temporary = output.with_suffix(".parquet.tmp")
        try:
            df.to_parquet(temporary, index=False, compression="snappy")
            if pq.read_metadata(temporary).num_rows != stats["rows"]:
                raise RuntimeError(f"Row count mismatch: {name}")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        records.append({"file": name, **stats})
        print(f"FILL {number}/{len(target_names)} {name}: {stats['changed_rows']:,} rows / "
              f"{stats['filled_cells']:,} cells "
              f"(pass1={stats['pass1']['filled_cells']:,}, "
              f"pass2={stats['pass2']['filled_cells']:,})", flush=True)
        del df

    report = {
        "status": "complete", "stage": "metadata_only_v21_two_pass",
        "dataset_group": selected,
        "donor_policy": "2025_training_plus_original_ranking_frozen_for_all_targets",
        "final_ranking_used_as_donor": False,
        "donor_rows_by_file": donor_counts,
        "pass1_usable_keys": len(first),
        "pass2_usable_keys": {f: len(v) for f, v in second.items()},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "pyarrow": pyarrow.__version__},
        "files": records,
    }
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(summary_path)
    print(f"\nCOMPLETE: metadata preprocessing ({len(target_names)} files).", flush=True)
    print(f"Output: {destination}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print("Next: core features + supplied external features.", flush=True)
    return report
