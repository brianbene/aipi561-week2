"""
Week 7: Cost Optimization & Feedback Loop Starter Template

Implement three systems:
1. CostAnalyzer - analyze and track query costs
2. OptimizationStrategy - optimize costs through caching, model selection, etc.
3. FeedbackLoop - collect and validate user corrections
"""

import json
import logging
import os
import statistics
from collections import Counter, OrderedDict
from typing import Dict, List, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CostAnalyzer:
    """Analyze and track query costs by component."""

    def __init__(self):
        """Initialize cost analyzer."""
        self.query_history: List[Dict[str, Any]] = []

    def record_query(self, query: Dict[str, Any]):
        """Record a query and its cost breakdown.

        Stores a normalized copy of the query dict with fields:
        - query_text, retrieval_cost, llm_cost, tool_cost, error_cost,
          total_cost, timestamp

        Missing numeric fields default to 0.0. If total_cost is not
        supplied, it is computed as the sum of the four cost components.
        """
        retrieval_cost = float(query.get("retrieval_cost", 0.0))
        llm_cost = float(query.get("llm_cost", 0.0))
        tool_cost = float(query.get("tool_cost", 0.0))
        error_cost = float(query.get("error_cost", 0.0))
        total_cost = query.get("total_cost")
        if total_cost is None:
            total_cost = retrieval_cost + llm_cost + tool_cost + error_cost
        else:
            total_cost = float(total_cost)

        entry = {
            "query_text": query.get("query_text", ""),
            "retrieval_cost": retrieval_cost,
            "llm_cost": llm_cost,
            "tool_cost": tool_cost,
            "error_cost": error_cost,
            "total_cost": total_cost,
            "timestamp": query.get("timestamp", datetime.now().isoformat()),
        }
        self.query_history.append(entry)
        logger.info(
            "Recorded query '%s' total_cost=$%.6f",
            entry["query_text"][:50],
            entry["total_cost"],
        )

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get breakdown of costs by component across all recorded queries."""
        retrieval_total = sum(q["retrieval_cost"] for q in self.query_history)
        llm_total = sum(q["llm_cost"] for q in self.query_history)
        tool_total = sum(q["tool_cost"] for q in self.query_history)
        error_total = sum(q["error_cost"] for q in self.query_history)
        total_daily = sum(q["total_cost"] for q in self.query_history)

        return {
            "retrieval_total": round(retrieval_total, 6),
            "llm_total": round(llm_total, 6),
            "tool_total": round(tool_total, 6),
            "error_total": round(error_total, 6),
            "total_daily": round(total_daily, 6),
            "query_count": len(self.query_history),
        }

    def identify_cost_spikes(self) -> List[Dict]:
        """Identify unusually expensive queries.

        Uses a simple statistical-outlier rule: a query is a "spike" if its
        total_cost exceeds mean + 2*stdev of all recorded total_costs.
        Requires at least 2 queries to compute a standard deviation; with
        fewer than that, returns an empty list (no baseline to compare to).
        """
        if len(self.query_history) < 2:
            return []

        costs = [q["total_cost"] for q in self.query_history]
        mean_cost = statistics.mean(costs)
        stdev_cost = statistics.stdev(costs)
        threshold = mean_cost + 2 * stdev_cost

        spikes = []
        for q in self.query_history:
            if q["total_cost"] > threshold:
                spikes.append(
                    {
                        "query_text": q["query_text"],
                        "total_cost": q["total_cost"],
                        "timestamp": q["timestamp"],
                        "mean_cost": round(mean_cost, 6),
                        "stdev_cost": round(stdev_cost, 6),
                        "threshold": round(threshold, 6),
                    }
                )
        return spikes


class OptimizationStrategy:
    """Optimize agent costs through multiple strategies."""

    def __init__(self, max_cache_size: int = 1000):
        """Initialize optimization strategy.

        Uses an OrderedDict as a simple LRU cache (bounded by max_cache_size)
        so memory does not grow unbounded over a long-running process.
        """
        self.cache: "OrderedDict[str, str]" = OrderedDict()
        self.max_cache_size = max_cache_size
        self.strategies_applied: List[str] = []
        self._cache_hits = 0
        self._cache_misses = 0
        self._model_selections = {"gemini-1.5-flash": 0, "gemini-2.5-pro": 0}
        self._retrieval_reductions: List[Dict[str, int]] = []
        self._compressions_applied = 0

    def apply_caching(self, query: str, response: str) -> tuple:
        """Cache query responses (LRU, bounded by max_cache_size).

        Returns:
            (is_cached_hit, response)
        """
        if query in self.cache:
            self.cache.move_to_end(query)
            self._cache_hits += 1
            if "caching" not in self.strategies_applied:
                self.strategies_applied.append("caching")
            return (True, self.cache[query])

        self._cache_misses += 1
        self.cache[query] = response
        self.cache.move_to_end(query)
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)
        if "caching" not in self.strategies_applied:
            self.strategies_applied.append("caching")
        return (False, response)

    def optimize_retrieval_count(self, num_docs: int) -> int:
        """Reduce number of documents retrieved (simple top-k reduction).

        Input 15 docs -> output 3 docs by default (reduce by 5x, floor of 1).
        """
        optimized = max(1, num_docs // 5)
        self._retrieval_reductions.append({"before": num_docs, "after": optimized})
        if "retrieval_reduction" not in self.strategies_applied:
            self.strategies_applied.append("retrieval_reduction")
        return optimized

    def select_model_by_complexity(self, query: str) -> str:
        """Choose cheaper model for simple queries, stronger model for complex ones.

        Heuristic: if the query contains an explicit complexity keyword
        (analyze, explain, compare, design), route to gemini-2.5-pro.
        Otherwise route to the cheaper gemini-1.5-flash.
        """
        complexity_words = ["analyze", "explain", "compare", "design"]
        query_lower = query.lower()
        if any(w in query_lower for w in complexity_words):
            model = "gemini-2.5-pro"
        else:
            model = "gemini-1.5-flash"

        self._model_selections[model] = self._model_selections.get(model, 0) + 1
        if "model_selection" not in self.strategies_applied:
            self.strategies_applied.append("model_selection")
        return model

    def enable_response_compression(self, response: str, max_sentences: int = 2) -> str:
        """Compress long responses while keeping essential info.

        Splits on sentence-ending punctuation and keeps only the first
        max_sentences sentences. Responses already at or under that length
        are returned unchanged.
        """
        import re

        sentences = re.split(r"(?<=[.!?])\s+", response.strip())
        sentences = [s for s in sentences if s]

        if len(sentences) <= max_sentences:
            return response

        compressed = " ".join(sentences[:max_sentences])
        self._compressions_applied += 1
        if "response_compression" not in self.strategies_applied:
            self.strategies_applied.append("response_compression")
        return compressed

    def get_optimization_impact(self) -> Dict[str, Any]:
        """Estimate cost savings from applied optimizations.

        Savings are estimated from observed behavior, not assumed:
        - Caching: % of all apply_caching() calls that were hits (a hit
          query incurs $0 LLM cost instead of a full call).
        - Model selection: % of select_model_by_complexity() calls that
          were routed to the cheaper flash model (assumed ~80% cheaper
          per call than pro, a commonly cited Gemini flash/pro price ratio).
        - Retrieval reduction: average % reduction in documents retrieved,
          which roughly tracks token cost reduction for retrieval-augmented
          prompts.
        """
        total_cache_calls = self._cache_hits + self._cache_misses
        cache_hit_rate = (
            self._cache_hits / total_cache_calls if total_cache_calls else 0.0
        )

        total_model_calls = sum(self._model_selections.values())
        flash_rate = (
            self._model_selections.get("gemini-1.5-flash", 0) / total_model_calls
            if total_model_calls
            else 0.0
        )
        model_selection_savings_pct = flash_rate * 80.0

        if self._retrieval_reductions:
            avg_reduction_pct = statistics.mean(
                (1 - r["after"] / r["before"]) * 100
                for r in self._retrieval_reductions
                if r["before"] > 0
            )
        else:
            avg_reduction_pct = 0.0

        breakdown = {
            "caching": {
                "cache_hit_rate_pct": round(cache_hit_rate * 100, 2),
                "calls_served_from_cache": self._cache_hits,
                "total_cache_calls": total_cache_calls,
            },
            "model_selection": {
                "pct_routed_to_cheaper_model": round(flash_rate * 100, 2),
                "estimated_savings_pct": round(model_selection_savings_pct, 2),
                "selections": dict(self._model_selections),
            },
            "retrieval_reduction": {
                "avg_doc_count_reduction_pct": round(avg_reduction_pct, 2),
                "reductions_applied": len(self._retrieval_reductions),
            },
            "response_compression": {
                "responses_compressed": self._compressions_applied,
            },
        }

        signals = [
            cache_hit_rate * 100,
            model_selection_savings_pct,
            avg_reduction_pct,
        ]
        total_savings_pct = round(sum(signals) / len(signals), 2) if signals else 0.0

        return {
            "total_savings_pct": total_savings_pct,
            "strategies_applied": list(self.strategies_applied),
            "breakdown": breakdown,
        }


class FeedbackLoop:
    """Collect and validate user corrections for continuous improvement."""

    def __init__(self):
        """Initialize feedback loop and role-based authority hierarchy."""
        self.corrections: List[Dict[str, Any]] = []
        self.authority = {
            "engineer": 1,
            "hr": 2,
            "finance": 2,
            "manager": 3,
            "executive": 4,
        }

    def submit_correction(
        self,
        original_query: str,
        original_answer: str,
        corrected_answer: str,
        user_role: str,
    ) -> Dict[str, Any]:
        """Submit a correction to the agent's answer.

        Validates at submission time:
        1. user_role must have sufficient authority (manager+, level >= 3)
        2. corrected_answer must be more detailed (longer) than original_answer

        The correction is always stored (for audit/analysis purposes), but
        is only marked accepted=True if it passes both checks.
        """
        authority_level = self.authority.get(user_role, 0)
        has_authority = authority_level >= 3
        is_more_detailed = len(corrected_answer) > len(original_answer)

        if not has_authority:
            accepted = False
            reason = (
                f"Role '{user_role}' (authority level {authority_level}) lacks "
                f"sufficient authority; manager-level (3) or above required."
            )
        elif not is_more_detailed:
            accepted = False
            reason = (
                "Corrected answer is not more detailed than the original "
                f"({len(corrected_answer)} chars vs {len(original_answer)} chars)."
            )
        else:
            accepted = True
            reason = "Correction accepted: sufficient authority and added detail."

        entry = {
            "original_query": original_query,
            "original_answer": original_answer,
            "corrected_answer": corrected_answer,
            "user_role": user_role,
            "accepted": accepted,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        self.corrections.append(entry)
        logger.info(
            "Correction submitted by role=%s accepted=%s", user_role, accepted
        )

        return {"accepted": accepted, "reason": reason}

    def validate_correction(self, index: int) -> bool:
        """Validate a stored correction is accurate.

        Re-checks (independent of what was stored at submission time):
        1. User role has sufficient authority (manager+, level >= 3)
        2. Correction is more detailed than original

        Args:
            index: index into corrections list

        Returns:
            True if correction is valid, False otherwise
        """
        if index < 0 or index >= len(self.corrections):
            return False

        correction = self.corrections[index]
        authority = self.authority.get(correction["user_role"], 0)
        return authority >= 3 and len(correction["corrected_answer"]) > len(
            correction["original_answer"]
        )

    def get_feedback_metrics(self) -> Dict[str, Any]:
        """Compute metrics on feedback quality."""
        total = len(self.corrections)
        if total == 0:
            return {
                "total_corrections": 0,
                "validation_rate": 0.0,
                "avg_correction_length": 0.0,
                "top_error_patterns": [],
            }

        valid_count = sum(
            1 for i in range(total) if self.validate_correction(i)
        )
        validation_rate = valid_count / total

        avg_correction_length = statistics.mean(
            len(c["corrected_answer"]) for c in self.corrections
        )

        pattern_counter = Counter(
            " ".join(c["original_answer"].split()[:4]) for c in self.corrections
        )
        top_error_patterns = [
            {"pattern": pattern, "count": count}
            for pattern, count in pattern_counter.most_common(5)
        ]

        return {
            "total_corrections": total,
            "validation_rate": round(validation_rate, 4),
            "avg_correction_length": round(avg_correction_length, 2),
            "top_error_patterns": top_error_patterns,
        }

    def save_corrections(self, path: str = "data/corrections.json"):
        """Persist corrections to disk so they survive a process restart."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.corrections, f, indent=2)
        logger.info("Saved %d corrections to %s", len(self.corrections), path)

    def load_corrections(self, path: str = "data/corrections.json"):
        """Load previously persisted corrections from disk."""
        with open(path, "r", encoding="utf-8") as f:
            self.corrections = json.load(f)
        logger.info("Loaded %d corrections from %s", len(self.corrections), path)


if __name__ == "__main__":

    print("=" * 70)
    print("TASK 1: Testing CostAnalyzer")
    print("=" * 70)

    analyzer = CostAnalyzer()

    normal_queries = [
        ("What is the travel policy?", 0.001, 0.004, 0.0, 0.0),
        ("Who can approve expenses over $5000?", 0.0015, 0.005, 0.001, 0.0),
        ("What is the NYC per diem rate?", 0.001, 0.003, 0.0, 0.0),
        ("What is the compensation policy?", 0.001, 0.004, 0.0, 0.0),
        ("What is Alice's role?", 0.0008, 0.0035, 0.0, 0.0),
        ("Look up employee John Smith.", 0.0012, 0.004, 0.0015, 0.0),
        ("What is the expense approval limit for an engineer?", 0.001, 0.0038, 0.0, 0.0),
        ("What is the total expenses for the Alpha project?", 0.0011, 0.0042, 0.0012, 0.0),
        ("What medical benefits are available to employees?", 0.0009, 0.0036, 0.0, 0.0),
        ("What is the per diem rate for Chicago?", 0.001, 0.0033, 0.0, 0.0),
    ]
    for text, r, l, t, e in normal_queries:
        analyzer.record_query(
            {
                "query_text": text,
                "retrieval_cost": r,
                "llm_cost": l,
                "tool_cost": t,
                "error_cost": e,
                "timestamp": datetime.now().isoformat(),
            }
        )

    analyzer.record_query(
        {
            "query_text": "Analyze and compare expense trends across all departments for the last 3 years",
            "retrieval_cost": 0.45,
            "llm_cost": 0.42,
            "tool_cost": 0.08,
            "error_cost": 0.05,
            "timestamp": datetime.now().isoformat(),
        }
    )

    breakdown = analyzer.get_cost_breakdown()
    print("\nCost breakdown across all recorded queries:")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")

    spikes = analyzer.identify_cost_spikes()
    print(f"\nCost spikes detected: {len(spikes)}")
    for s in spikes:
        print(
            f"  SPIKE: '{s['query_text'][:60]}' cost=${s['total_cost']:.4f} "
            f"(threshold=${s['threshold']:.4f}, mean=${s['mean_cost']:.4f}, "
            f"stdev=${s['stdev_cost']:.4f})"
        )

    print("\n" + "=" * 70)
    print("TASK 2: Testing OptimizationStrategy")
    print("=" * 70)

    optimizer = OptimizationStrategy(max_cache_size=5)

    print("\n-- Caching --")
    hit1, resp1 = optimizer.apply_caching("What is the travel policy?", "Answer A")
    hit2, resp2 = optimizer.apply_caching("What is the travel policy?", "Answer B (ignored, cache wins)")
    print(f"  First call  -> is_cached_hit={hit1}, response='{resp1}'")
    print(f"  Second call -> is_cached_hit={hit2}, response='{resp2}'")
    assert hit1 is False and hit2 is True and resp1 == resp2
    print("  [OK] Cache returns identical response on repeat query.")

    print("\n-- Model selection by complexity --")
    simple_q = "What is the per diem rate for NYC?"
    complex_q = "Analyze and compare expense trends across departments"
    simple_model = optimizer.select_model_by_complexity(simple_q)
    complex_model = optimizer.select_model_by_complexity(complex_q)
    print(f"  Simple query  -> '{simple_q}' routed to: {simple_model}")
    print(f"  Complex query -> '{complex_q}' routed to: {complex_model}")
    assert simple_model == "gemini-1.5-flash"
    assert complex_model == "gemini-2.5-pro"
    print("  [OK] Simple queries use the cheaper model, complex queries use the stronger one.")

    print("\n-- Retrieval count optimization --")
    reduced = optimizer.optimize_retrieval_count(15)
    print(f"  15 documents requested -> optimized to {reduced} documents")
    assert reduced == 3

    print("\n-- Response compression --")
    long_response = (
        "The travel policy allows business class for flights over 8 hours. "
        "Economy class is standard for shorter flights. "
        "All bookings must be made through the corporate travel portal. "
        "Receipts must be submitted within 30 days of travel."
    )
    compressed = optimizer.enable_response_compression(long_response, max_sentences=2)
    print(f"  Original ({len(long_response)} chars): {long_response}")
    print(f"  Compressed ({len(compressed)} chars): {compressed}")
    assert len(compressed) < len(long_response)

    print("\n-- Optimization impact summary --")
    impact = optimizer.get_optimization_impact()
    print(json.dumps(impact, indent=2))

    print("\n" + "=" * 70)
    print("TASK 3: Testing FeedbackLoop")
    print("=" * 70)

    feedback = FeedbackLoop()

    test_corrections = [
        (
            "What is the travel policy for flights over 8 hours?",
            "There is no specific policy for 8+ hour flights.",
            "Employees can book business class for flights over 8 hours with manager approval.",
            "manager",
        ),
        (
            "What is the SSN format?",
            "SSNs are 9 digits.",
            "SSNs follow the format XXX-XX-XXXX.",
            "engineer",
        ),
        (
            "What is the compensation policy?",
            "Compensation is reviewed annually based on performance, market data, and budget, with detailed bands per level.",
            "Reviewed yearly.",
            "executive",
        ),
        (
            "What medical benefits are available?",
            "Some benefits exist.",
            "Employees have access to medical, dental, and vision coverage, plus an HSA match up to $1000/year.",
            "hr",
        ),
        (
            "What is the per diem rate for international travel?",
            "It varies.",
            "International per diem rates are set per destination city following the GSA foreign rate table, updated quarterly.",
            "executive",
        ),
    ]

    for query, orig, corr, role in test_corrections:
        result = feedback.submit_correction(query, orig, corr, role)
        print(f"  Role={role:<10} accepted={result['accepted']!s:<5} reason={result['reason']}")

    print("\n-- Feedback metrics --")
    metrics = feedback.get_feedback_metrics()
    print(json.dumps(metrics, indent=2))

    print("\n-- Persisting corrections to disk --")
    feedback.save_corrections("data/corrections.json")
    reload_test = FeedbackLoop()
    reload_test.load_corrections("data/corrections.json")
    print(f"  Reloaded {len(reload_test.corrections)} corrections from data/corrections.json")
    assert len(reload_test.corrections) == len(feedback.corrections)
    print("  [OK] Corrections persist and reload correctly.")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
