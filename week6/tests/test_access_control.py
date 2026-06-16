import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week6.app.access_control import AccessController, RateLimiter, CostEnforcer

POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "data", "access_control.json")


# ── AccessController tests ───────────────────────────────────────────────────

def test_access_controller_loads_policy():
    """Policy file loads without error and roles are populated."""
    ac = AccessController(POLICY_PATH)
    assert "engineer" in ac.roles
    assert "hr" in ac.roles
    assert "finance" in ac.roles


def test_can_view_document_allowed():
    """Engineer can view an api_docs document."""
    ac = AccessController(POLICY_PATH)
    doc = {"id": "doc1", "title": "API Guide", "category": "api_docs"}
    assert ac.can_view_document("engineer", doc) is True


def test_can_view_document_denied():
    """Engineer cannot view an hr_policies document."""
    ac = AccessController(POLICY_PATH)
    doc = {"id": "doc2", "title": "Salary Bands", "category": "hr_policies"}
    assert ac.can_view_document("engineer", doc) is False


def test_can_view_document_hr_sees_all():
    """HR role can view any document category."""
    ac = AccessController(POLICY_PATH)
    doc = {"id": "doc3", "title": "Budget Report", "category": "budgets"}
    assert ac.can_view_document("hr", doc) is True


def test_can_view_field_non_sensitive():
    """Any role can view a non-sensitive field like name."""
    ac = AccessController(POLICY_PATH)
    assert ac.can_view_field("engineer", "name") is True
    assert ac.can_view_field("engineer", "email") is True


def test_can_view_field_salary_denied_for_engineer():
    """Engineer cannot view the salary field."""
    ac = AccessController(POLICY_PATH)
    assert ac.can_view_field("engineer", "salary") is False


def test_can_view_field_salary_allowed_for_hr():
    """HR can view the salary field."""
    ac = AccessController(POLICY_PATH)
    assert ac.can_view_field("hr", "salary") is True


def test_can_view_field_ssn_denied_for_engineer():
    """Engineer cannot view SSN."""
    ac = AccessController(POLICY_PATH)
    assert ac.can_view_field("engineer", "ssn") is False


def test_redact_response_removes_ssn():
    """SSN pattern is redacted for non-HR roles."""
    ac = AccessController(POLICY_PATH)
    response = "Employee SSN is 123-45-6789 per records."
    redacted = ac.redact_response("engineer", response)
    assert "123-45-6789" not in redacted
    assert "[REDACTED-SSN]" in redacted


def test_redact_response_keeps_ssn_for_hr():
    """SSN is NOT redacted for HR role."""
    ac = AccessController(POLICY_PATH)
    response = "Employee SSN is 123-45-6789 per records."
    redacted = ac.redact_response("hr", response)
    assert "123-45-6789" in redacted


def test_redact_response_removes_salary():
    """Salary pattern is redacted for engineer role."""
    ac = AccessController(POLICY_PATH)
    response = "The employee salary: $95,000 per year."
    redacted = ac.redact_response("engineer", response)
    assert "95,000" not in redacted


def test_audit_log_records_access():
    """log_access appends an entry to the audit log."""
    ac = AccessController(POLICY_PATH)
    ac.log_access("engineer", "doc1", allowed=True, user_id="user_001")
    ac.log_access("engineer", "salary_data", allowed=False, user_id="user_001")
    log = ac.get_audit_log()
    assert len(log) == 2
    assert log[0]["action"] == "ALLOW"
    assert log[1]["action"] == "DENY"
    assert log[1]["role"] == "engineer"


def test_filter_documents_removes_restricted():
    """filter_documents returns only docs the role can access."""
    ac = AccessController(POLICY_PATH)
    docs = [
        {"id": "d1", "title": "API Docs", "category": "api_docs"},
        {"id": "d2", "title": "HR Policy", "category": "hr_policies"},
        {"id": "d3", "title": "Deploy Guide", "category": "deployment_guides"},
    ]
    result = ac.filter_documents("engineer", docs)
    ids = [d["id"] for d in result]
    assert "d1" in ids
    assert "d3" in ids
    assert "d2" not in ids


def test_denial_count():
    """get_denial_count returns correct number of denied entries."""
    ac = AccessController(POLICY_PATH)
    ac.log_access("engineer", "hr_policies", allowed=False)
    ac.log_access("engineer", "api_docs", allowed=True)
    ac.log_access("engineer", "budgets", allowed=False)
    assert ac.get_denial_count() == 2


# ── RateLimiter tests ────────────────────────────────────────────────────────

def test_rate_limiter_allows_within_limit():
    """First 3 queries are allowed when limit is 3."""
    limiter = RateLimiter(max_queries_per_minute=3)
    assert limiter.is_allowed("user1") is True
    assert limiter.is_allowed("user1") is True
    assert limiter.is_allowed("user1") is True


def test_rate_limiter_blocks_at_limit():
    """4th query is blocked when limit is 3."""
    limiter = RateLimiter(max_queries_per_minute=3)
    limiter.is_allowed("user2")
    limiter.is_allowed("user2")
    limiter.is_allowed("user2")
    assert limiter.is_allowed("user2") is False


def test_rate_limiter_independent_users():
    """Hitting limit for user1 does not affect user2."""
    limiter = RateLimiter(max_queries_per_minute=2)
    limiter.is_allowed("userA")
    limiter.is_allowed("userA")
    assert limiter.is_allowed("userA") is False
    assert limiter.is_allowed("userB") is True


def test_rate_limiter_remaining_queries():
    """get_remaining_queries returns correct count."""
    limiter = RateLimiter(max_queries_per_minute=5)
    limiter.is_allowed("user3")
    limiter.is_allowed("user3")
    assert limiter.get_remaining_queries("user3") == 3


# ── CostEnforcer tests ───────────────────────────────────────────────────────

def test_cost_enforcer_allows_within_budget():
    """Query within budget is allowed."""
    enforcer = CostEnforcer()
    assert enforcer.can_afford_query("user1", 50.0) is True


def test_cost_enforcer_blocks_when_exceeded():
    """Query that would exceed budget is blocked."""
    enforcer = CostEnforcer()
    enforcer.add_cost("user1", "engineer", 50.0)
    # engineer budget is $100; $50 spent + $51 estimated = $101 > $100
    assert enforcer.can_afford_query("user1", 51.0) is False


def test_cost_enforcer_tracks_spending():
    """add_cost accumulates correctly."""
    enforcer = CostEnforcer()
    enforcer.add_cost("user2", "hr", 30.0)
    enforcer.add_cost("user2", "hr", 20.0)
    remaining = enforcer.get_budget_remaining("user2")
    assert remaining == 150.0  # hr budget $200 - $50 spent


def test_cost_enforcer_role_budgets_differ():
    """Engineer and manager have different budgets."""
    enforcer = CostEnforcer()
    enforcer.add_cost("eng1", "engineer", 0.0)
    enforcer.add_cost("mgr1", "manager", 0.0)
    assert enforcer.get_budget_remaining("eng1") == 100.0
    assert enforcer.get_budget_remaining("mgr1") == 500.0


def test_cost_enforcer_exact_budget_boundary():
    """Spending exactly the budget amount is allowed; one cent over is not."""
    enforcer = CostEnforcer()
    enforcer.add_cost("user3", "engineer", 99.99)
    assert enforcer.can_afford_query("user3", 0.01) is True   # exactly $100
    assert enforcer.can_afford_query("user3", 0.02) is False  # $100.01


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
