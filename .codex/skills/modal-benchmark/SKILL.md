---
name: modal-benchmark
description: Run the canonical paired clean-machine Modal CPU benchmark for this SuperMarioBros-Nes-turbo repo. Requires a candidate ref; if the user omits the baseline, use the latest local `main` commit. The launcher compares exact local git archives without switching branches, runs multiple paired same-host Modal replicas, and reports robust paired speedup for the first four Super Mario Bros NES Level1 states mixed across vector lanes.
---

# Modal Benchmark

## Contract

This skill requires a candidate ref. If the user does not mention a baseline,
the baseline is always the latest commit resolved by the local `main` branch at
run time.

```text
candidate_ref
baseline_ref candidate_ref
```

When two refs are supplied, the baseline ref is always the first ref and the
candidate ref is always the second ref. If the user invokes this skill without a
candidate ref, ask for the missing candidate instead of running a one-sided
benchmark.

Do not switch local branches for the comparison. The launcher resolves each ref
locally with `git rev-parse`, creates exact `git archive` source snapshots for
those commits, uploads both archives to Modal, and benchmarks those snapshots in
the same Modal job. Local dirty or untracked files are not included unless the
user commits them and passes that commit/ref.

The canonical workload remains:

- `replicas=7`
- `pairs_per_replica=7`
- `warmup_pairs_per_replica=1`, discarded from decision statistics but kept in
  the artifact
- `num_envs=16`
- `steps=20000`
- `frame_skip=4`
- `frame_stack=4`
- grayscale, crop top 32, resize 84x84
- action `noop`
- states `Level1-1,Level1-2,Level1-3,Level1-4`
- observation contract `obs_shape=(16, 4, 84, 84)`, `obs_dtype=uint8`

## Workflow

Run the paired Modal launcher from the repository root:

```bash
modal run scripts/modal_compare_sps.py \
  --candidate-ref CANDIDATE_REF \
  --output-json artifacts/benchmarks/modal-compare-YYYY-MM-DD-HHMM.json
```

Add `--baseline-ref BASELINE_REF` only when the user explicitly names a
baseline other than `main`.

Use the current date in the artifact name. If that path already exists, add a
short time suffix.

The launcher defaults to
`--replicas 7 --pairs-per-replica 7 --warmup-pairs-per-replica 1 --steps 20000`.
Each Modal replica runs on one Modal host, builds both refs, installs them into
separate per-ref virtualenvs, then alternates baseline and candidate samples on
that same host:

```text
warmup pair: baseline sample 1
warmup pair: candidate sample 1
candidate sample 2
baseline sample 2
...
```

The final decision metric is paired and host-normalized, not absolute:

```text
replica_ratio = median(candidate_sample_i / baseline_sample_i) for measured pairs on one replica
robust_paired_speedup = median(replica_ratio_j)
```

For optimization acceptance, prefer candidates with:

- robust paired speedup `> 1.10`
- 95% bootstrap lower confidence bound over replica medians `> 1.05`
- candidate faster on at least `5/7` replica medians
- no obvious metadata or correctness mismatch

Treat one-sided absolute Modal SPS and pooled absolute SPS stdev as
smoke-test/diagnostic information only. Modal host heterogeneity can dominate
absolute SPS variance; use the reported variance decomposition to separate
within-replica noise from between-replica host spread.

The launcher also maintains a local commit-stats cache at
`artifacts/benchmarks/modal-compare-stats-cache.json` by default. Cache entries
are keyed by commit SHA, git archive hash, benchmark config, ROM/state hashes,
and SHA-256 hashes of both `scripts/modal_compare_sps.py` and this skill file.
Changing either the launcher or `.codex/skills/modal-benchmark/SKILL.md` makes
old entries stale because the benchmark context hash changes. This cache records
per-commit absolute stats from completed runs; it does not replace the paired
same-host Modal samples used for acceptance decisions.

Use `--stats-cache-json PATH` to choose a different cache file. Use
`--no-write-stats-cache` when a diagnostic run should not update the cache.

The launcher uploads local ROM bytes plus the four stable-retro `.state` files
at runtime. State files resolve from an explicit `--state-dir`,
`SUPERMARIOBROSNES_FASTENV_STATE_DIR`, an installed `stable_retro` package, or
the sibling `stable-retro-turbo` checkout.

If Modal access is unavailable, report the blocker plainly.

## Reporting

After the Modal run completes, read the saved JSON artifact and report:

- whether it worked, and that Modal compared exact `git archive` snapshots
  without switching local branches
- the absolute artifact path
- baseline ref/SHA and candidate ref/SHA
- replica count, pairs per replica, warmup pair count, measured pair count, and
  total pair count
- raw measured-pair speedup median ratio, mean ratio, median gain percent, mean
  gain percent
- robust replica-median speedup ratio, 95% bootstrap confidence interval,
  faster-replica count, and launcher verdict
- baseline and candidate median/mean/stdev/min/max env steps/sec
- per-replica ratio summaries and CPU metadata
- variance decomposition for baseline, candidate, and ratio: pooled stdev,
  within-replica stdev, between-replica mean stdev, and replica means
- estimated Modal compute cost, including pricing source and any caveat from
  the JSON artifact
- stats-cache path, benchmark context hash, launcher hash, skill hash, and
  whether each commit entry was already present before the run
- the Modal run URL if command output includes one
- whether launcher/skill/docs changes were made

End the final answer with a Markdown results table. Keep the table compact and
put the acceptance verdict immediately above it. Use this shape:

```text
| Metric | Baseline | Candidate | Paired / Notes |
| --- | ---: | ---: | --- |
| Ref | BASELINE_REF | CANDIDATE_REF | baseline defaults to main if omitted |
| SHA | BASELINE_SHA12 | CANDIDATE_SHA12 | exact git archive snapshots |
| Median SPS | BASELINE_MEDIAN | CANDIDATE_MEDIAN | median ratio RATIO, gain GAIN% |
| Mean SPS | BASELINE_MEAN | CANDIDATE_MEAN | mean ratio RATIO, gain GAIN% |
| Robust Ratio |  |  | replica-median RATIO, CI95 LOW-HIGH, faster FASTER/REPLICAS |
| Stdev SPS | BASELINE_STDEV | CANDIDATE_STDEV | MEASURED_PAIR_COUNT measured, WARMUP_PAIR_COUNT warmup discarded |
| Range SPS | BASELINE_MIN-BASELINE_MAX | CANDIDATE_MIN-CANDIDATE_MAX | CPU: CPU_MODEL, CPUs AFFINITY_COUNT |
| Est. Cost |  |  | TOTAL_USD from REPLICA_WALL_TIME_S replica-seconds |
| Cache | HIT_OR_MISS | HIT_OR_MISS | CONTEXT_SHA12 |
```

Add or remove rows only when a field is unavailable or a user asks for a
different table.

Use this extraction helper after the run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/benchmarks/modal-compare-YYYY-MM-DD-HHMM.json")
data = json.loads(path.read_text())
summary = data["summary"]
ratio = summary["candidate_over_baseline"]
robust = summary["replica_median_candidate_over_baseline"]
ci = summary["replica_median_bootstrap_ci"]
decision = summary["decision"]
baseline = summary["baseline_env_steps_per_sec"]
candidate = summary["candidate_env_steps_per_sec"]
print("path", path.resolve())
print("baseline", data["refs"]["baseline"]["ref"], data["refs"]["baseline"]["sha"])
print("candidate", data["refs"]["candidate"]["ref"], data["refs"]["candidate"]["sha"])
print(
    "paired_speedup",
    "median_ratio", ratio["median"],
    "mean_ratio", ratio["mean"],
    "median_gain_pct", summary["paired_speedup_pct_median"],
    "mean_gain_pct", summary["paired_speedup_pct_mean"],
    "pairs", summary["pair_count"],
    "warmup_pairs", summary.get("warmup_pair_count"),
    "total_pairs", summary.get("total_pair_count"),
    "replicas", summary["replica_count"],
)
print(
    "robust_paired_speedup",
    "replica_median_ratio", robust["median"],
    "replica_median_mean", robust["mean"],
    "ci95_lower", ci["lower"],
    "ci95_upper", ci["upper"],
    "candidate_faster_replicas", decision["candidate_faster_replica_medians"],
    "replicas", decision["replica_count"],
    "verdict", decision["verdict"],
)
print("baseline_sps", baseline)
print("candidate_sps", candidate)
for name, decomposition in summary.get("variance_decomposition", {}).items():
    print(
        "variance_decomposition",
        name,
        "pooled_stdev", decomposition["pooled"]["stdev"],
        "within_replica_stdev", decomposition["within_replica_stdev"],
        "between_replica_mean_stdev", decomposition["between_replica_mean_stdev"],
        "replica_means", decomposition["replica_means"],
    )
cost = data.get("cost_estimate", {})
print(
    "estimated_modal_compute_cost",
    "total_usd", cost.get("total_cost_usd"),
    "cpu_usd", cost.get("cpu_cost_usd"),
    "memory_usd", cost.get("memory_cost_usd"),
    "replica_wall_time_s", cost.get("replica_wall_time_s"),
    "pricing_source", cost.get("source"),
    "note", cost.get("note"),
)
cache = data.get("cache", {})
print("stats_cache_path", cache.get("stats_cache_path"))
print("benchmark_context_sha256", cache.get("benchmark_context_sha256"))
for name, meta in cache.get("tool_hashes", {}).items():
    print("tool_hash", name, meta.get("path"), meta.get("sha256"))
for label, entry in cache.get("entries", {}).items():
    print(
        "cache_entry",
        label,
        "key", entry.get("cache_key"),
        "hit_before_run", entry.get("hit_before_run"),
    )
for replica in data["replicas"]:
    r = replica["summary"]["candidate_over_baseline"]
    m = replica["modal"]
    print(
        "replica",
        replica["replica_index"],
        "median_ratio", r["median"],
        "mean_ratio", r["mean"],
        "measured_pairs", replica["summary"].get("measured_pair_count"),
        "warmup_pairs", replica["summary"].get("warmup_pair_count"),
        "cpu_model", m.get("cpu_model"),
        "affinity_cpu_count", m.get("affinity_cpu_count"),
    )
PY
```

Run `git status --short` before the final answer so changed launcher/docs files
are not hidden.
