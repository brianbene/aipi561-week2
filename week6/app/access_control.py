import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any


class AccessController:
    """Enforce role-based access control for documents and fields."""

    def __init__(self, access_policy_path: str):
        with open(access_policy_path, "r", encoding="utf-8-sig") as f:
            self.policy = json.load(f)
        self.roles = self.policy.get("roles", {})
        self.sensitive_fields = self.policy.get("sensitive_fields", {})
        self.document_categories = self.policy.get("document_categories", {})
        self.audit_log: List[Dict] = []

    def can_view_document(self, role: str, document: Dict) -> bool:
        """Check if role can view this document based on its category."""
        role_config = self.roles.get(role, {})
        allowed_docs = role_config.get("permissions", {}).get("documents", [])

        # Role with "all" access can see every document
        if "all" in allowed_docs:
            return True

        # Check document category against allowed list
        doc_category = document.get("category", "")
        if doc_category in allowed_docs:
            return True

        # Check document_categories mapping
        for category, allowed_roles in self.document_categories.items():
            if doc_category == category and role in allowed_roles:
                return True

        return False

    def can_view_field(self, role: str, field_name: str) -> bool:
        """Check if role can view this field."""
        # If field is not sensitive, everyone can see it
        if field_name not in self.sensitive_fields:
            return True

        # Check if role is in the visibility list for this sensitive field
        visibility = self.sensitive_fields[field_name].get("visibility", [])
        return role in visibility

    def redact_response(self, role: str, response: str) -> str:
        """Scan response text for sensitive field values and redact them."""
        redacted = response

        # Redact SSN patterns (e.g. 123-45-6789)
        if not self.can_view_field(role, "ssn"):
            redacted = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]", redacted)

        # Redact salary patterns (e.g. $120,000 or $120000 or salary: 120000)
        if not self.can_view_field(role, "salary"):
            redacted = re.sub(
                r"(?i)(salary[:\s]+\$?[\d,]+|\$[\d,]{4,}(?:\.\d{2})?(?:\s*/\s*(?:year|yr|annual))?)",
                "[REDACTED-SALARY]",
                redacted
            )

        # Redact medical info keywords
        if not self.can_view_field(role, "medical_info"):
            redacted = re.sub(
                r"(?i)(medical[_\s]info[:\s]+\S+|diagnosis[:\s]+\S+|condition[:\s]+\S+)",
                "[REDACTED-MEDICAL]",
                redacted
            )

        # Redact bonus patterns
        if not self.can_view_field(role, "bonus"):
            redacted = re.sub(
                r"(?i)(bonus[:\s]+\$?[\d,]+)",
                "[REDACTED-BONUS]",
                redacted
            )

        return redacted

    def log_access(self, role: str, resource: str, allowed: bool, user_id: str = "unknown"):
        """Append an audit log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": user_id,
            "role": role,
            "resource": resource,
            "allowed": allowed,
            "action": "ALLOW" if allowed else "DENY"
        }
        self.audit_log.append(entry)

    def filter_documents(self, role: str, documents: List[Dict]) -> List[Dict]:
        """Return only documents this role is permitted to view."""
        filtered = []
        for doc in documents:
            allowed = self.can_view_document(role, doc)
            self.log_access(role, doc.get("id", doc.get("title", "unknown")), allowed)
            if allowed:
                filtered.append(doc)
        return filtered

    def get_audit_log(self) -> List[Dict]:
        """Return the full audit log."""
        return self.audit_log

    def get_denial_count(self) -> int:
        """Return total number of denied access attempts."""
        return sum(1 for entry in self.audit_log if not entry["allowed"])


class RateLimiter:
    """Rate limit queries per user per minute."""

    def __init__(self, max_queries_per_minute: int = 30):
        self.max_queries = max_queries_per_minute
        # Maps user_id -> list of timestamps (float) of recent queries
        self.query_times: Dict[str, List[float]] = {}

    def _clean_old_queries(self, user_id: str):
        """Remove query timestamps older than 60 seconds."""
        now = time.time()
        cutoff = now - 60.0
        if user_id in self.query_times:
            self.query_times[user_id] = [
                t for t in self.query_times[user_id] if t > cutoff
            ]

    def is_allowed(self, user_id: str) -> bool:
        """Return True if the user can make another query this minute."""
        self._clean_old_queries(user_id)
        current_count = len(self.query_times.get(user_id, []))
        if current_count >= self.max_queries:
            return False
        # Record this query attempt
        if user_id not in self.query_times:
            self.query_times[user_id] = []
        self.query_times[user_id].append(time.time())
        return True

    def get_remaining_queries(self, user_id: str) -> int:
        """Return how many more queries this user can make in the current window."""
        self._clean_old_queries(user_id)
        used = len(self.query_times.get(user_id, []))
        return max(0, self.max_queries - used)


class CostEnforcer:
    """Enforce monthly budget limits per role."""

    # Monthly budgets by role in USD
    ROLE_BUDGETS = {
        "engineer": 100.0,
        "manager": 500.0,
        "hr": 200.0,
        "finance": 500.0,
        "executive": 1000.0,
    }
    DEFAULT_BUDGET = 100.0

    def __init__(self, policy_path: str = None):
        # Maps user_id -> {"role": str, "spent": float}
        self.user_spending: Dict[str, Dict] = {}
        # Optionally load overrides from policy file
        if policy_path:
            try:
                with open(policy_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    overrides = data.get("budgets", {})
                    self.ROLE_BUDGETS = {**self.ROLE_BUDGETS, **overrides}
            except Exception:
                pass

    def _get_budget(self, role: str) -> float:
        return self.ROLE_BUDGETS.get(role, self.DEFAULT_BUDGET)

    def add_cost(self, user_id: str, role: str, cost: float):
        """Record spending for a user."""
        if user_id not in self.user_spending:
            self.user_spending[user_id] = {"role": role, "spent": 0.0}
        self.user_spending[user_id]["spent"] += cost
        self.user_spending[user_id]["role"] = role

    def can_afford_query(self, user_id: str, estimated_cost: float) -> bool:
        """Return True if the user has enough budget remaining."""
        role = self.user_spending.get(user_id, {}).get("role", "engineer")
        budget = self._get_budget(role)
        spent = self.user_spending.get(user_id, {}).get("spent", 0.0)
        return (spent + estimated_cost) <= budget

    def get_budget_remaining(self, user_id: str) -> float:
        """Return remaining budget for the user."""
        role = self.user_spending.get(user_id, {}).get("role", "engineer")
        budget = self._get_budget(role)
        spent = self.user_spending.get(user_id, {}).get("spent", 0.0)
        return max(0.0, budget - spent)

    def get_spending_summary(self) -> Dict[str, Any]:
        """Return spending summary across all users."""
        return {
            uid: {
                "role": info["role"],
                "spent": info["spent"],
                "budget": self._get_budget(info["role"]),
                "remaining": self._get_budget(info["role"]) - info["spent"]
            }
            for uid, info in self.user_spending.items()
        }
