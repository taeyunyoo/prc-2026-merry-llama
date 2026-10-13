"""Stage 2: 29 v21 core features and 17 external features for 14 datasets.

No targets are used to build features. Target-derived medians, neighborhood,
runway-relative wind, arrival features and model fitting are later stages.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import core
from . import external
from ..data import ProjectPaths, dataset_names, training_names

ADDED = core.ADDED + external.ADDED
assert len(ADDED) == len(set(ADDED)) == 46
CONTRACT = "prc2026.base_features.v1"
MARKER = "step02_complete.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def implementation_fingerprint():
    files = [Path(__file__), Path(core.__file__), Path(external.__file__),
             Path(__file__).parents[1] / "data.py"]
    return {path.name: digest(path) for path in files}


def source_record(path, root):
    try:
        name = path.relative_to(root).as_posix()
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": digest(path), "size_bytes": path.stat().st_size}


def check_preprocessing(input_dir, names):
    marker = input_dir / "preprocessing_summary.json"
    if not marker.is_file():
        raise FileNotFoundError(
            f"Metadata completion report missing: {marker}. "
            "Use the completed metadata_filled output from scripts/preproc.py -s all."
        )
    report = json.loads(marker.read_text(encoding="utf-8"))
    records = report.get("files", [])
    if (report.get("status") != "complete" or report.get("dataset_group") != "all"
            or len(records) != len(names) or {r["file"] for r in records} != set(names)):
        raise ValueError("Metadata completion report does not describe all 14 datasets")
    for record in records:
        path = input_dir / record["file"]
        if not path.is_file() or pq.read_metadata(path).num_rows != record["rows"]:
            raise ValueError(f"Metadata output differs from its completion report: {path}")
    return marker


def validate_base_outputs(root=None):
    """Verify the complete stage, its source files and all output contents.

    Later stages can call this before consuming base_features. It never writes.
    Hashes use content rather than modification times, allowing project relocation.
    """
    project = ProjectPaths.from_root(root)
    marker = project.base_features_dir / MARKER
    report = json.loads(marker.read_text(encoding="utf-8"))
    if (report.get("status") != "complete" or report.get("contract") != CONTRACT
            or report.get("added_features") != ADDED
            or report.get("implementation") != implementation_fingerprint()):
        raise ValueError("Base-feature completion contract or implementation differs")
    records = report.get("files", [])
    if len(records) != 14 or {r["file"] for r in records} != set(dataset_names()):
        raise ValueError("Base-feature completion report must contain all 14 datasets")
    inputs = report.get("inputs", [])
    if len(inputs) != 18:
        raise ValueError("Base-feature report must identify 14 inputs, 3 resources and metadata report")
    for record in inputs:
        path = project.root / record["path"]
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"Base-feature source changed or missing: {path}")
    for record in records:
        path = project.base_features_dir / record["file"]
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"Base-feature output changed or missing: {path}")
    return report


def canonical_map(names, required):
    lookup = {}
    for name in names:
        key = name.lower()
        if key in lookup:
            raise ValueError(f"Ambiguous column names: {name}, {lookup[key]}")
        lookup[key] = name
    missing = [name for name in required if name.lower() not in lookup]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    return {name: lookup[name.lower()] for name in required}


def canonical_frame(table, required):
    mapping = canonical_map(table.column_names, required)
    return table.select(list(mapping.values())).to_pandas().rename(
        columns={value: key for key, value in mapping.items()}
    ).reset_index(drop=True)


def read_canonical(path, required):
    mapping = canonical_map(pq.read_schema(path).names, required)
    return canonical_frame(core.read_table(path, list(mapping.values())), required)


def resolve_external(root, kind, explicit=None):
    """Select one file by required schema under data/external, never by mtime."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    directory = root / "data" / "external"
    if not directory.is_dir():
        raise FileNotFoundError(f"External data directory not found: {directory}")
    required = {
        "weather": {"station", "valid", "tmpf", "sknt", "vsby", "wxcodes", "metar"},
        "reference": {"airport", "date", *external.REF},
        "distance": {"ADEP", "ADES", "distance_km"},
    }[kind]
    suffix = ".parquet" if kind == "weather" else ".csv"
    matches = []
    for path in sorted(directory.rglob(f"*{suffix}")):
        if kind == "weather":
            names = pq.read_schema(path).names
        else:
            names = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
        if required.issubset(names):
            matches.append(path)
    if len(matches) != 1:
        details = "\n".join(str(path) for path in matches) or "(none)"
        raise ValueError(
            f"Expected exactly one {kind} source under {directory}; found {len(matches)}.\n"
            f"{details}\nSpecify --{kind} with the intended file path."
        )
    return matches[0].resolve()


def movement_context(path):
    cols = [core.ID, core.TIME, "PHASE_mvt", "ADEP_mvt", "ADEP_flt",
            "RUNWAY_mvt", "WK_TBL_CAT_flt"]
    d = read_canonical(path, cols)
    d = d.loc[core.dep_mask(d)].copy()
    d["ADEP"] = core.clean(d["ADEP_mvt"]).fillna(core.clean(d["ADEP_flt"]))
    d["runway"] = core.clean(d["RUNWAY_mvt"])
    d["heavy"] = core.clean(d["WK_TBL_CAT_flt"]).isin(["H", "J", "HEAVY", "SUPER"])
    d[core.TIME] = external.dates(d[core.TIME])
    if core.clean(d[core.ID]).isna().any():
        raise ValueError(f"{path.name}: missing departure movement IDs")
    if d[core.ID].duplicated().any():
        raise ValueError(f"{path.name}: duplicate departure movement IDs")
    unknown = set(d["ADEP"].dropna()) - set(core.TIMEZONES)
    if unknown:
        raise ValueError(f"{path.name}: no timezone configured for {sorted(unknown)}")
    return d[[core.ID, core.TIME, "ADEP", "runway", "heavy"]].reset_index(drop=True)


def periods(kind):
    ranges = {
        "training": [("2025-01-01", "2026-01-01")],
        "ranking": [("2026-01-01", "2026-02-01"), ("2026-07-01", "2026-08-01")],
        "final_ranking": [("2026-01-01", "2026-03-01"), ("2026-06-01", "2026-08-01")],
    }[kind]
    return [(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")) for a, b in ranges]


def valid_context(frame):
    # Preserve invalid-key rows in the output, but do not use them as donors.
    return frame.loc[frame[[core.TIME, "ADEP"]].notna().all(axis=1)].reset_index(drop=True)


def check_context(frame, kind):
    if frame[core.ID].duplicated().any():
        raise ValueError(f"{kind}: duplicate departure IDs across source files")
    covered = pd.Series(False, index=frame.index)
    for start, end in periods(kind):
        covered |= frame[core.TIME].ge(start) & frame[core.TIME].lt(end)
    # A small timestamp spillover must not cause source rows to be deleted.
    return {"departures": len(frame),
            "missing_airport": int(frame.ADEP.isna().sum()),
            "missing_time": int(frame[core.TIME].isna().sum()),
            "outside_declared_periods": int((~covered & frame[core.TIME].notna()).sum())}


def external_keys(d):
    return pd.DataFrame({
        "ADEP": core.clean(d.ADEP), "ADES": core.clean(d.ADES),
        "time": external.dates(d[core.TIME]),
    }, index=d.index)


def match_valid_external(d, weather, reference, distance):
    result = pd.DataFrame(np.nan, index=d.index, columns=external.ADDED)
    valid = d.ADEP.notna() & d.time.notna()
    result["wx_metar_available_flag"] = 0.0
    if valid.any():
        matched = external.match_external(d.loc[valid], weather, reference, distance)
        result.loc[valid, external.ADDED] = matched
    return result


def atomic_save(raw, features, destination):
    full = features.reindex(pd.RangeIndex(raw.num_rows))
    table = raw
    for name in ADDED:
        dtype = pa.string() if name in core.PAIRS else pa.float64()
        table = table.append_column(name, pa.array(full[name], type=dtype, from_pandas=True))
    temp = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        pq.write_table(table, temp, compression="snappy")
        saved = core.read_table(temp)
        if not saved.select(raw.column_names).equals(raw):
            raise RuntimeError(f"Original columns or row order changed: {destination.name}")
        if not saved.equals(table):
            raise RuntimeError(f"Feature serialization mismatch: {destination.name}")
        del saved
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def run_base_features(root=None, *, weather_path=None, reference_path=None, distance_path=None,
                 overwrite=False):
    project = ProjectPaths.from_root(root)
    root = project.root
    input_dir = project.metadata_filled_dir
    output_dir = project.base_features_dir
    names = dataset_names()
    paths = [input_dir / name for name in names]
    metadata_marker = check_preprocessing(input_dir, names)
    for zone in set(core.TIMEZONES.values()):
        ZoneInfo(zone)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        schema = pq.read_schema(path)
        canonical_map(schema.names, core.REQUIRED)
        collision = set(x.lower() for x in schema.names) & set(x.lower() for x in ADDED)
        if collision:
            raise ValueError(f"{path.name}: derived columns already exist: {sorted(collision)}")
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not overwrite:
        marker = output_dir / MARKER
        if not marker.is_file():
            raise FileExistsError(
                f"Incomplete or unverified output exists: {existing[0]}. "
                "Use --overwrite to rebuild only the base-feature stage."
            )
        print("Checking completed base-feature outputs and source fingerprints...", flush=True)
        report = validate_base_outputs(root)
        for kind, explicit in [("weather", weather_path), ("reference", reference_path),
                               ("distance", distance_path)]:
            requested = resolve_external(root, kind, explicit)
            previous = Path(root / report["external_sources"][kind]).resolve()
            if requested != previous:
                raise ValueError(f"Different {kind} source requested; use --overwrite to rebuild")
        print(f"Output: {output_dir}", flush=True)
        print("UNCHANGED: verified existing base-feature outputs; no files rewritten.", flush=True)
        print("COMPLETE: base/external feature generation (14 files, 46 added columns).", flush=True)
        return report

    wx_path = resolve_external(root, "weather", weather_path)
    ref_path = resolve_external(root, "reference", reference_path)
    dist_path = resolve_external(root, "distance", distance_path)
    sources = paths + [wx_path, ref_path, dist_path, metadata_marker]
    stamps = {path: core.stamp(path) for path in sources}
    print(f"Input: {input_dir}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    for kind, path in [("Weather", wx_path), ("Reference", ref_path), ("Distance", dist_path)]:
        print(f"{kind}: {path}", flush=True)
    weather = external.load_weather(wx_path)
    reference = external.load_reference(ref_path)
    distance = external.load_distance(dist_path)
    print("Recording source fingerprints...", flush=True)
    source_records = [source_record(path, root) for path in sources]

    # Preflight all reference keys before any parquet is written.
    contexts = []
    for i, path in enumerate(paths, 1):
        print(f"[Preflight {i}/14] {path.name}", flush=True)
        contexts.append(movement_context(path))
        d = read_canonical(path, ["PHASE_mvt", "ADEP_mvt", "ADEP_flt", core.TIME])
        d = d.loc[core.dep_mask(d)].copy()
        d["ADEP"] = core.clean(d.ADEP_mvt).fillna(core.clean(d.ADEP_flt))
        d["time"] = external.dates(d[core.TIME])
        valid = d.ADEP.notna() & d.time.notna()
        try:
            external.reference_values(d.loc[valid], reference)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        if path.name in {"ranking.parquet", "final_ranking.parquet"}:
            target = read_canonical(path, ["PHASE_mvt", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"])
            mask = core.dep_mask(target)
            for column in ["TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"]:
                if core.clean(target.loc[mask, column]).notna().any():
                    raise ValueError(f"{path.name}: unexpected DEP values in {column}")
    train = pd.concat(contexts[:12], ignore_index=True)
    rank, final = contexts[12], contexts[13]
    del contexts
    context_report = {kind: check_context(frame, kind) for kind, frame in
                      [("training", train), ("ranking", rank), ("final_ranking", final)]}
    for label, evaluation in [("ranking", rank), ("final_ranking", final)]:
        if train[core.ID].isin(evaluation[core.ID]).any():
            raise ValueError(f"Training and {label} share departure IDs")
    print("Computing departure counts and same-runway heavy/super counts...", flush=True)
    train_valid, rank_valid, final_valid = map(valid_context, [train, rank, final])
    train_counts = core.airport_counts(train_valid, periods("training"))
    rank_counts = core.airport_counts(rank_valid, periods("ranking"))
    final_counts = core.airport_counts(final_valid, periods("final_ranking"))
    # Preserve the legacy training/ranking donor universe exactly.
    legacy_wake = core.wake_counts(pd.concat([train_valid, rank_valid], ignore_index=True))
    final_wake = core.wake_counts(pd.concat([train_valid, final_valid], ignore_index=True))
    overlap_count = int(rank[core.ID].isin(final[core.ID]).sum())
    del train, rank, final, train_valid, rank_valid, final_valid
    gc.collect()

    for path in sources:
        if core.stamp(path) != stamps[path]:
            raise RuntimeError(f"Input changed during preflight: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = output_dir / MARKER
    complete.unlink(missing_ok=True)
    summaries, missing = [], []
    for i, path in enumerate(paths, 1):
        print(f"[Build {i}/14] {path.name}", flush=True)
        if core.stamp(path) != stamps[path]:
            raise RuntimeError(f"Input changed: {path}")
        raw = core.read_table(path)
        d = canonical_frame(raw, core.REQUIRED)
        dep = d.loc[core.dep_mask(d)]
        if path.name == "final_ranking.parquet":
            counts, wake = final_counts, final_wake
        elif path.name == "ranking.parquet":
            counts, wake = rank_counts, legacy_wake
        else:
            counts, wake = train_counts, legacy_wake
        f = core.derive(dep, counts, wake)
        keys = external_keys(pd.concat([dep, f[list(core.PAIRS)]], axis=1))
        ext = match_valid_external(keys, weather, reference, distance)
        f = pd.concat([f, ext], axis=1)[ADDED]
        numeric = [name for name in ADDED if name not in core.PAIRS]
        if np.isinf(f[numeric].to_numpy(dtype=float)).any():
            raise ValueError(f"{path.name}: infinite feature values")
        if core.stamp(path) != stamps[path]:
            raise RuntimeError(f"Input changed while being read: {path}")
        atomic_save(raw, f, output_dir / path.name)
        row = {"file": path.name, "rows": raw.num_rows, "departures": len(dep),
               "added_columns": len(ADDED),
               "weather_unmatched": int(ext.wx_metar_available_flag.eq(0).sum()),
               "reference_unmatched": int(ext[external.REF].isna().any(axis=1).sum()),
               "distance_missing": int(ext.distance_km.isna().sum()),
               "missing_departure_airport": int(keys.ADEP.isna().sum()),
               "missing_movement_time": int(keys.time.isna().sum())}
        row["sha256"] = digest(output_dir / path.name)
        summaries.append(row)
        missing.extend({"file": path.name, "feature": name, "missing_departures": int(count),
                        "departures": len(dep)} for name, count in f.isna().sum().items())
        print(f"  Saved {raw.num_rows:,} rows | DEP {len(dep):,} | +46 features | "
              f"weather unmatched {row['weather_unmatched']:,} | "
              f"distance missing {row['distance_missing']:,}", flush=True)
        # A partial run keeps its per-file report; complete marker is written last.
        pd.DataFrame(summaries).drop(columns="sha256").to_csv(output_dir / "step02_file_summary.csv", index=False)
        del raw, d, dep, f, ext, keys
        gc.collect()
    for path in sources:
        if core.stamp(path) != stamps[path]:
            raise RuntimeError(f"Input changed during feature generation: {path}")
    pd.DataFrame(missing).to_csv(output_dir / "step02_missingness.csv", index=False)
    payload = {
        "stage": 2, "contract": CONTRACT, "status": "complete",
        "files": summaries, "added_features": ADDED,
        "implementation": implementation_fingerprint(),
        "contexts": context_report, "ranking_final_overlapping_departure_ids": overlap_count,
        "count_policy": "Separate training, ranking and final_ranking; no ranking/final concatenation",
        "wake_policy": "Training/ranking: training+ranking. Final: training+final_ranking.",
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "pyarrow": pa.__version__},
        "inputs": source_records,
        "external_sources": {kind: record["path"] for kind, record in
                             zip(("weather", "reference", "distance"), source_records[14:17])},
    }
    write_json(complete, payload)
    print("\n" + pd.DataFrame(summaries).drop(columns="sha256").to_string(index=False), flush=True)
    print(f"\nOutput: {output_dir}", flush=True)
    print("COMPLETE: base/external feature generation (14 files, 46 added columns).", flush=True)
    return payload
