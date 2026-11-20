from __future__ import annotations

import heapq
import time

import pandas as pd

from .config import MemoryConfig
from .constructive import concept_neighbors, evidence_decision, initial_candidates
from .evaluation import decision_metrics
from .tasks import ConceptLibrary, concept_to_code
from .trajectory import EpisodeTrajectory


def evaluate_classical_search(
    trajectories: list[EpisodeTrajectory],
    config: MemoryConfig,
    library: ConceptLibrary,
    budget: int,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostics = []
    for trajectory in trajectories:
        started = time.perf_counter()
        initial = initial_candidates(trajectory, library)
        visited = set(initial)
        expanded = set()
        generated_unique = set()
        generated_instances = 0
        evidence_score_calls = 0
        evidence_scored_instances = 0
        expansions = 0

        def scores(candidates: set[int]) -> dict[int, float]:
            nonlocal evidence_score_calls, evidence_scored_instances
            if not candidates:
                return {}
            evidence_score_calls += 1
            evidence_scored_instances += len(candidates)
            decision = evidence_decision(trajectory, set(candidates), config, "_score", library)
            return {code: decision.posterior.get(code, 0.0) for code in candidates}

        if len(visited) > budget:
            ranked = scores(visited)
            visited = set(sorted(visited, key=ranked.get, reverse=True)[:budget])
        if strategy == "greedy_search":
            while len(visited) < budget:
                remaining = visited - expanded
                if not remaining:
                    break
                ranked = scores(visited)
                parents = sorted(remaining, key=ranked.get, reverse=True)[:4]
                pair_pool = set(sorted(visited, key=ranked.get, reverse=True)[:8])
                new = set()
                for parent in parents:
                    neighbors = concept_neighbors(parent, pair_pool, library)
                    generated_unique.update(neighbors)
                    generated_instances += len(neighbors)
                    new |= neighbors
                    expanded.add(parent)
                    expansions += 1
                new -= visited
                if not new:
                    break
                new_scores = scores(new)
                available = budget - len(visited)
                visited |= set(sorted(new, key=new_scores.get, reverse=True)[:available])
        elif strategy == "beam_search":
            ranked = scores(visited)
            frontier = set(sorted(visited, key=ranked.get, reverse=True)[:4])
            while frontier and len(visited) < budget:
                pair_pool = set(sorted(frontier, key=ranked.get, reverse=True)[:8])
                new = set()
                for parent in frontier:
                    neighbors = concept_neighbors(parent, pair_pool, library)
                    generated_unique.update(neighbors)
                    generated_instances += len(neighbors)
                    new |= neighbors
                    expansions += 1
                expanded |= frontier
                new -= visited
                if not new:
                    break
                new_scores = scores(new)
                available = budget - len(visited)
                accepted = set(sorted(new, key=new_scores.get, reverse=True)[:available])
                visited |= accepted
                ranked = scores(frontier | accepted)
                frontier = set(sorted(frontier | accepted, key=ranked.get, reverse=True)[:4])
        elif strategy == "astar_search":
            initial_scores = scores(visited)
            heap = [(1.0 - initial_scores[code], 0, code) for code in visited]
            heapq.heapify(heap)
            best_cost = {code: 0 for code in visited}
            while heap and len(visited) < budget:
                _, cost, parent = heapq.heappop(heap)
                if parent in expanded or cost != best_cost.get(parent):
                    continue
                expanded.add(parent)
                expansions += 1
                pair_pool = set(sorted(visited, key=initial_scores.get, reverse=True)[:8])
                neighbors = concept_neighbors(parent, pair_pool, library)
                generated_unique.update(neighbors)
                generated_instances += len(neighbors)
                new = neighbors - visited
                if not new:
                    continue
                new_scores = scores(new)
                available = budget - len(visited)
                accepted = set(sorted(new, key=new_scores.get, reverse=True)[:available])
                for code in accepted:
                    next_cost = cost + 1
                    if next_cost >= best_cost.get(code, float("inf")):
                        continue
                    visited.add(code)
                    best_cost[code] = next_cost
                    initial_scores[code] = new_scores[code]
                    heapq.heappush(heap, (next_cost + 1.0 - new_scores[code], next_cost, code))
        else:
            raise ValueError(strategy)
        search_seconds = time.perf_counter() - started
        evidence_started = time.perf_counter()
        decision = evidence_decision(trajectory, visited, config, strategy, library)
        evidence_seconds = time.perf_counter() - evidence_started
        rows.append(decision_metrics(decision, trajectory))
        diagnostics.append(
            {
                "split": trajectory.episode.split,
                "episode": trajectory.episode.index,
                "method": strategy,
                "candidate_recall": float(concept_to_code(trajectory.episode.concept) in visited),
                "candidate_count": len(visited),
                "initial_candidate_count": len(initial),
                "generated_unique_candidates": len(generated_unique),
                "generated_candidate_instances": generated_instances,
                "controller_score_calls": 0,
                "controller_scored_instances": 0,
                "controller_scored_unique": 0,
                "policy_scored_instances": 0,
                "ranker_scored_candidates": 0,
                "evidence_score_calls": evidence_score_calls,
                "evidence_scored_instances": evidence_scored_instances,
                "evidence_scored_candidates": len(visited),
                "expansions": expansions,
                "search_budget": budget,
                "search_seconds": search_seconds,
                "evidence_seconds": evidence_seconds,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)
