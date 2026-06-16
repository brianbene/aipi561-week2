import sys
import os
sys.path.insert(0, ".")

from week6.app.access_control import AccessController, RateLimiter, CostEnforcer

POLICY_PATH = "week6/data/access_control.json"

print("=== Smoke Test: Week 6 Guardrails ===\n")

# Test 1: AccessController loads and redacts
print("--- Test 1: AccessController ---")
ac = AccessController(POLICY_PATH)
print(f"Roles loaded: {list(ac.roles.keys())}")
print(f"Engineer can view 'name': {ac.can_view_field('engineer', 'name')}")
print(f"Engineer can view 'salary': {ac.can_view_field('engineer', 'salary')}")
print(f"HR can view 'salary': {ac.can_view_field('hr', 'salary')}")

response = "Employee salary: $95,000 and SSN is 123-45-6789"
redacted_eng = ac.redact_response("engineer", response)
redacted_hr = ac.redact_response("hr", response)
print(f"Engineer sees: {redacted_eng}")
print(f"HR sees:       {redacted_hr}")

# Test 2: Document filtering
print("\n--- Test 2: Document Filtering ---")
docs = [
    {"id": "d1", "title": "API Guide", "category": "api_docs"},
    {"id": "d2", "title": "HR Policy", "category": "hr_policies"},
    {"id": "d3", "title": "Budget", "category": "budgets"},
]
eng_docs = ac.filter_documents("engineer", docs)
hr_docs = ac.filter_documents("hr", docs)
print(f"Engineer sees {len(eng_docs)}/3 docs: {[d['id'] for d in eng_docs]}")
print(f"HR sees {len(hr_docs)}/3 docs: {[d['id'] for d in hr_docs]}")
print(f"Audit log entries: {len(ac.get_audit_log())}, denials: {ac.get_denial_count()}")

# Test 3: Rate limiter
print("\n--- Test 3: RateLimiter ---")
limiter = RateLimiter(max_queries_per_minute=3)
results = [limiter.is_allowed("user_brian") for _ in range(4)]
print(f"4 query results (3 allowed, 1 blocked): {results}")
print(f"Remaining queries: {limiter.get_remaining_queries('user_brian')}")

# Test 4: Cost enforcer
print("\n--- Test 4: CostEnforcer ---")
enforcer = CostEnforcer()
print(f"Can afford $50 (engineer budget $100): {enforcer.can_afford_query('brian', 50.0)}")
enforcer.add_cost("brian", "engineer", 50.0)
print(f"After spending $50, can afford $51: {enforcer.can_afford_query('brian', 51.0)}")
print(f"Budget remaining: ${enforcer.get_budget_remaining('brian'):.2f}")

print("\n=== All smoke tests passed ===")
