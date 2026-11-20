from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreammem.config import load_config


TASKS = {
    "dcoder": (17, 29, 43),
    "relational": (17, 29, 43),
    "advanced": (17,),
    "large": (17,),
    "natural": (17,),
    "noisy10": (17,),
    "noisy25": (17,),
    "support2": (17,),
    "support4": (17,),
    "support16": (17,),
    "support32": (17,),
    "nanodiff": (17,),
}
TRAINING = {
    "dcoder": "dcoder_training",
    "relational": "relational_training",
    "advanced": "advanced_training",
    "large": "large_training",
    "natural": "natural_training",
    "nanodiff": "nanodiff_training",
}


def require_frame(path: Path, rows: int, required: tuple[str, ...]) -> pd.DataFrame:
    assert path.exists(), f"Missing {path}"
    frame = pd.read_csv(path)
    assert len(frame) == rows, f"{path}: {len(frame)} != {rows}"
    for column in required:
        assert column in frame, f"Missing column {column} in {path}"
        assert not frame[column].isna().any(), f"NaN in {path}:{column}"
    return frame


EVALUATION_METHODS = (
    "final", "token_stability", "semantic_stability", "bag_of_states",
    "revision_no_merge", "merge_only", "revision_dynamics", "evidence_states",
    "evidence_guided", "symbolic", "closure_search", "random_search",
    "random_search_matched", "learned_search",
)
SEARCH_METHODS = ("closure_search", "random_search", "random_search_matched", "learned_search")
AUDIT_METHODS = (
    "closure_search", "random_search", "random_search_matched", "learned_search",
    "learned_ranker_full", "learned_ranker_initial", "learned_search_policy_rank",
    "evidence_guided", "symbolic",
)
ABLATION_METHODS = (
    "learned_search", "learned_search_atoms", "learned_search_no_merge",
    "learned_search_no_evidence_features", "learned_search_no_trajectory_features",
    "learned_search_logistic", "learned_search_mlp", "random_search_matched",
    "closure_search", "greedy_search", "beam_search", "astar_search",
    "learned_ranker_full", "learned_ranker_initial", "learned_search_policy_rank",
    "evidence_guided", "symbolic",
)
DECODER_METHODS = (
    "final", "token_stability", "semantic_stability", "bag_of_states",
    "revision_no_merge", "merge_only", "revision_dynamics", "evidence_states",
    "evidence_guided", "symbolic",
)
METRIC_COLUMNS = (
    "concept_accuracy", "query_accuracy", "query_nll", "true_code", "true_size",
)


def require_grid(
    frame: pd.DataFrame,
    splits: list[str],
    episodes: int,
    methods: tuple[str, ...],
    path: Path,
) -> None:
    keys = ["split", "episode", "method"]
    assert not frame.duplicated(keys).any(), f"Duplicate result key in {path}"
    expected = pd.MultiIndex.from_product([splits, range(episodes), methods], names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    assert set(actual) == set(expected), f"Incomplete split/episode/method grid in {path}"


def validate_seed(root: Path, task: str, seed: int) -> dict:
    config = load_config(root / "configs" / f"{task}.yaml")
    episodes = config.eval.episodes
    splits = len(config.eval.splits)
    base = root / "outputs" / "constructive_evaluation" / task / f"seed_{seed}"
    evaluation = base / "evaluation"
    summary = require_frame(
        evaluation / "summary.csv",
        splits * len(EVALUATION_METHODS),
        ("split", "method", "concept_accuracy", "query_accuracy", "query_nll"),
    )
    assert set(map(tuple, summary[["split", "method"]].to_numpy())) == {
        (split, method) for split in config.eval.splits for method in EVALUATION_METHODS
    }
    results = require_frame(
        evaluation / "results_all.csv",
        splits * episodes * len(EVALUATION_METHODS),
        ("split", "episode", "method", *METRIC_COLUMNS, "seed"),
    )
    require_grid(results, config.eval.splits, episodes, EVALUATION_METHODS, evaluation / "results_all.csv")
    assert set(results["seed"]) == {seed}
    oracle = require_frame(
        evaluation / "oracle_decoder.csv",
        splits * episodes,
        ("split", "episode", "query_accuracy", "query_nll"),
    )
    assert not oracle.duplicated(["split", "episode"]).any()
    assert set(map(tuple, oracle[["split", "episode"]].to_numpy())) == {
        (split, episode) for split in config.eval.splits for episode in range(episodes)
    }
    diagnostics = require_frame(
        evaluation / "search_diagnostics.csv",
        splits * episodes * len(SEARCH_METHODS),
        ("split", "episode", "method", "candidate_count", "candidate_recall"),
    )
    require_grid(
        diagnostics, config.eval.splits, episodes, SEARCH_METHODS,
        evaluation / "search_diagnostics.csv",
    )
    for name in ("intervals.csv", "concept_cluster_intervals.csv", "paired_comparisons.csv"):
        path = evaluation / name
        assert path.exists() and path.stat().st_size > 0, f"Missing {path}"
    for split in config.eval.splits:
        trajectory = evaluation / "trajectories" / f"{split}.pt"
        assert trajectory.exists() and trajectory.stat().st_size > 0, f"Missing {trajectory}"
    audit_results = require_frame(
        evaluation / "audit" / "audit_results.csv",
        splits * episodes * len(AUDIT_METHODS),
        ("split", "episode", "method", *METRIC_COLUMNS, "seed"),
    )
    require_grid(
        audit_results, config.eval.splits, episodes, AUDIT_METHODS,
        evaluation / "audit" / "audit_results.csv",
    )
    assert set(audit_results["seed"]) == {seed}
    for method in (
        "learned_search", "random_search", "random_search_matched", "closure_search",
        "evidence_guided", "symbolic",
    ):
        columns = ["split", "episode", *METRIC_COLUMNS]
        expected = results[results["method"] == method].sort_values(["split", "episode"])[columns].reset_index(drop=True)
        actual = audit_results[audit_results["method"] == method].sort_values(["split", "episode"])[columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            expected, actual, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    audit_costs = require_frame(
        evaluation / "audit" / "audit_costs.csv",
        splits * episodes * len(AUDIT_METHODS),
        (
            "split", "episode", "method", "dllm_forward_passes", "candidate_recall",
            "generated_unique_candidates", "generated_candidate_instances",
            "controller_scored_instances", "evidence_scored_candidates",
        ),
    )
    require_grid(
        audit_costs, config.eval.splits, episodes, AUDIT_METHODS,
        evaluation / "audit" / "audit_costs.csv",
    )
    ablation_results = require_frame(
        evaluation / "ablations" / "ablation_results.csv",
        splits * episodes * len(ABLATION_METHODS),
        ("split", "episode", "method", *METRIC_COLUMNS),
    )
    require_grid(
        ablation_results, config.eval.splits, episodes, ABLATION_METHODS,
        evaluation / "ablations" / "ablation_results.csv",
    )
    ablation_costs = require_frame(
        evaluation / "ablations" / "ablation_costs.csv",
        splits * episodes * len(ABLATION_METHODS),
        ("split", "episode", "method", "search_seconds", "evidence_seconds"),
    )
    require_grid(
        ablation_costs, config.eval.splits, episodes, ABLATION_METHODS,
        evaluation / "ablations" / "ablation_costs.csv",
    )
    runtime = json.loads((evaluation / "runtime.json").read_text(encoding="utf-8"))
    expected_passes = (
        config.trajectory.samples
        * config.trajectory.steps
        * len(config.trajectory.support_stages)
    )
    assert runtime["forward_passes_per_episode"] == expected_passes
    assert runtime["episodes_per_split"] == episodes
    assert runtime["episode_seed"] == 17100
    if task in TRAINING:
        training = root / "outputs" / TRAINING[task] / f"seed_{seed}"
        metadata = json.loads((training / "training.json").read_text(encoding="utf-8"))
        assert metadata["steps_completed"] == config.train.steps == 800
        assert (training / "best" / "experiment_metadata.json").exists()
    return {
        "task": task,
        "seed": seed,
        "splits": splits,
        "episodes_per_split": episodes,
        "result_rows": len(results),
        "forward_passes": expected_passes,
        "complete": True,
    }


def validate_decoders(root: Path) -> list[dict]:
    rows = []
    splits = ["seen", "unseen", "compositional"]
    for mode, passes in (("ar", 294), ("mlm", 18)):
        path = root / "outputs" / "decoder_evaluation" / "dcoder" / "seed_17" / mode
        results = require_frame(
            path / "results_all.csv",
            3 * 256 * len(DECODER_METHODS),
            ("split", "episode", "method", *METRIC_COLUMNS, "decoder"),
        )
        require_grid(results, splits, 256, DECODER_METHODS, path / "results_all.csv")
        assert set(results["decoder"]) == {mode}
        summary = require_frame(
            path / "summary.csv",
            3 * len(DECODER_METHODS),
            ("split", "method", "concept_accuracy", "query_accuracy", "query_nll"),
        )
        assert set(map(tuple, summary[["split", "method"]].to_numpy())) == {
            (split, method) for split in splits for method in DECODER_METHODS
        }
        runtime = json.loads((path / "runtime.json").read_text(encoding="utf-8"))
        assert runtime["forward_passes_per_episode"] == passes
        assert runtime["episode_seed"] == 17100
        for split in splits:
            trajectory = path / "trajectories" / f"{mode}_{split}.pt"
            assert trajectory.exists() and trajectory.stat().st_size > 0, f"Missing {trajectory}"
        rows.append({"task": f"decoder_{mode}", "seed": 17, "complete": True, "forward_passes": passes})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = []
    failures = []
    for task, seeds in TASKS.items():
        for seed in seeds:
            try:
                rows.append(validate_seed(root, task, seed))
            except (AssertionError, FileNotFoundError, json.JSONDecodeError) as error:
                failures.append({"task": task, "seed": seed, "error": str(error)})
    try:
        rows.extend(validate_decoders(root))
    except (AssertionError, FileNotFoundError, json.JSONDecodeError) as error:
        failures.append({"task": "decoders", "seed": 17, "error": str(error)})
    if args.final:
        assert not failures, failures
        for document in ("PAPER.md", "README.md"):
            text = (root / document).read_text(encoding="utf-8")
            assert not any(marker in text for marker in ("TODO", "TBD", "PENDING")), document
        statistics = root / "outputs" / "final_statistics"
        for name in (
            "method_metrics.csv", "ablation_metrics.csv", "concept_metrics.csv",
            "difficulty_metrics.csv", "paired_effects.csv", "compute_costs.csv",
            "all_episode_results.csv", "all_ablation_episode_results.csv",
            "paper_tables.md", "search_cost_tradeoff.png",
        ):
            path = statistics / name
            assert path.exists() and path.stat().st_size > 0, f"Missing {path}"
        metadata = json.loads((statistics / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["final"] and metadata["runs"] == len(rows)
        for task in TASKS:
            for suffix in ("concept_accuracy.png", "concept_difficulty.png"):
                path = statistics / f"{task}_{suffix}"
                assert path.exists() and path.stat().st_size > 0, f"Missing {path}"
        destination = root / "outputs" / "final_validation"
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(destination / "inventory.csv", index=False)
        (destination / "validation.json").write_text(
            json.dumps({"complete": True, "runs": rows}, indent=2), encoding="utf-8"
        )
    print(json.dumps({"complete_runs": len(rows), "incomplete_runs": failures}, indent=2))
    if args.final:
        print(f"Validated {len(rows)} complete runs")


if __name__ == "__main__":
    main()
