from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MemoryConfig
from .memory import MemoryFormer, _concept_slots, _error
from .semantics import INVALID_CONCEPT
from .tasks import code_to_concept, concept_label, concept_to_code
from .trajectory import EpisodeTrajectory


FEATURE_GROUPS = {
    "final": ["final_frequency", "final_confidence", "concept_size"],
    "token_stability": ["final_frequency", "confidence", "concept_size", "token_stability"],
    "semantic_stability": [
        "final_frequency",
        "confidence",
        "concept_size",
        "semantic_stability",
    ],
    "bag_of_states": ["frequency", "confidence", "concept_size"],
    "revision_dynamics": [
        "final_frequency",
        "frequency",
        "confidence",
        "concept_size",
        "token_stability",
        "semantic_stability",
        "revision_score",
    ],
}


def candidate_rows(trajectory: EpisodeTrajectory) -> pd.DataFrame:
    former = MemoryFormer(MemoryConfig())
    occurrence = former._occurrences(trajectory)
    counts: Counter[int] = occurrence["counts"]
    final_counts: Counter[int] = occurrence["final_counts"]
    confidence_sum = occurrence["confidence_sum"]
    final_confidence_sum = occurrence["final_confidence_sum"]
    total = max(int(occurrence["total"]), 1)
    token_probabilities = former._token_stability(trajectory)
    semantic = former._semantic_stability(trajectory)
    revisions, generated = former._revision_scores(trajectory)
    validation_start = max(stage.support_count for stage in trajectory.stages)
    true_code = concept_to_code(trajectory.episode.concept)
    rows = []
    for code in set(counts) | generated:
        concept = code_to_concept(code)
        slots = _concept_slots(code)
        token_score = float(
            np.mean(
                [
                    token_probabilities[position].get(token, 0.0)
                    for position, token in enumerate(slots)
                ]
            )
        )
        query_utility = float(
            np.mean(
                [
                    concept_label(concept, query.attributes) == query.label
                    for query in trajectory.episode.queries
                ]
            )
        )
        rows.append(
            {
                "episode": trajectory.episode.index,
                "code": code,
                "correct": float(code == true_code),
                "query_utility": query_utility,
                "frequency": counts[code] / total,
                "final_frequency": final_counts[code] / max(sum(final_counts.values()), 1),
                "confidence": confidence_sum[code] / max(counts[code], 1),
                "final_confidence": final_confidence_sum[code] / max(final_counts[code], 1),
                "concept_size": len(concept),
                "validation_error": _error(
                    code,
                    trajectory.episode,
                    validation_start,
                ),
                "token_stability": token_score,
                "semantic_stability": semantic.get(code, 0.0),
                "revision_score": revisions.get(code, 0.0),
            }
        )
    return pd.DataFrame(rows)


def run_probe(
    train_trajectories: list[EpisodeTrajectory],
    test_trajectories: list[EpisodeTrajectory],
) -> pd.DataFrame:
    train = pd.concat([candidate_rows(item) for item in train_trajectories], ignore_index=True)
    test = pd.concat([candidate_rows(item) for item in test_trajectories], ignore_index=True)
    if train.empty or test.empty:
        return pd.DataFrame(
            [
                {
                    "features": name,
                    "train_candidates": len(train),
                    "test_candidates": len(test),
                    "roc_auc": float("nan"),
                    "average_precision": float("nan"),
                    "utility_rmse": float("nan"),
                    "utility_rmse_gain": float("nan"),
                }
                for name in FEATURE_GROUPS
            ]
        )
    rows = []
    for name, features in FEATURE_GROUPS.items():
        row: dict[str, float | int | str] = {
            "features": name,
            "train_candidates": len(train),
            "test_candidates": len(test),
        }
        if train["correct"].nunique() > 1 and test["correct"].nunique() > 1:
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=1000),
            )
            classifier.fit(train[features], train["correct"])
            probability = classifier.predict_proba(test[features])[:, 1]
            row["roc_auc"] = roc_auc_score(test["correct"], probability)
            row["average_precision"] = average_precision_score(test["correct"], probability)
        else:
            row["roc_auc"] = float("nan")
            row["average_precision"] = float("nan")
        regressor = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        regressor.fit(train[features], train["query_utility"])
        prediction = regressor.predict(test[features])
        row["utility_rmse"] = float(np.sqrt(mean_squared_error(test["query_utility"], prediction)))
        baseline = float(np.sqrt(mean_squared_error(test["query_utility"], np.full(len(test), train["query_utility"].mean()))))
        row["utility_rmse_gain"] = baseline - row["utility_rmse"]
        rows.append(row)
    return pd.DataFrame(rows)
