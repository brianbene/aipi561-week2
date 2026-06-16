import json
import sqlite3
import os
import sys
import logging
from typing import Dict, Any

import google.genai as genai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from week6.app.access_control import AccessController, RateLimiter, CostEnforcer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

ACCESS_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "week6", "data", "access_control.json"
)


class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, **kwargs) -> str:
        raise NotImplementedError


class EmployeeLookupTool(Tool):
    def __init__(self, db_path: str):
        super().__init__("employee_lookup", "Find employee information by name or ID")
        self.db_path = db_path

    def execute(self, employee_name: str = None, employee_id: str = None) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if employee_id:
                cursor.execute("SELECT id, name, department, role FROM employees WHERE id = ?", (employee_id,))
            elif employee_name:
                cursor.execute("SELECT id, name, department, role FROM employees WHERE name LIKE ?", (f"%{employee_name}%",))
            else:
                return "Error: provide employee_name or employee_id"
            rows = cursor.fetchall()
            conn.close()
            if not rows:
                return "Employee not found"
            return json.dumps([dict(r) for r in rows], indent=2)
        except Exception as e:
            logger.error(f"Employee lookup error: {e}")
            return f"Error: {str(e)}"


class PolicySearchTool(Tool):
    def __init__(self, documents_path: str):
        super().__init__("policy_search", "Search policy documents by keyword or topic")
        self.documents = []
        try:
            with open(documents_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            logger.info(f"Loaded {len(self.documents)} policy documents")
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")

    def execute(self, query: str = None, keyword: str = None, keywords: str = None, limit: int = 3) -> str:
        try:
            query = query or keyword or keywords or ""
            if not self.documents:
                return "No policy documents available"
            query_lower = query.lower()
            scored = []
            for doc in self.documents:
                text = (doc.get("title", "") + " " + doc.get("content", "") + " " + doc.get("category", "")).lower()
                score = sum(1 for word in query_lower.split() if word in text)
                if score > 0:
                    scored.append((score, doc))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, doc in scored[:limit]:
                results.append({
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "category": doc.get("category"),
                    "snippet": doc.get("content", "")[:500]
                })
            if not results:
                return f"No policy documents found matching '{query}'"
            return json.dumps(results, indent=2)
        except Exception as e:
            logger.error(f"Policy search error: {e}")
            return f"Error: {str(e)}"


class ExpenseQueryTool(Tool):
    def __init__(self, db_path: str):
        super().__init__("expense_query", "Query expense approval limits and per-diem rates")
        self.db_path = db_path

    def execute(self, query_type: str, **kwargs) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if query_type == "approval_limit":
                role = kwargs.get("role", "engineer")
                cursor.execute("SELECT approval_limit FROM expense_policies WHERE role = ?", (role,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    return f"Approval limit for {role}: ${row[0]:,.2f}"
                return f"No approval limit found for role: {role}"
            elif query_type == "per_diem":
                location = kwargs.get("location", "")
                cursor.execute("SELECT daily_limit FROM per_diem WHERE location = ?", (location,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    return f"Per diem for {location}: ${row[0]:.2f}/day"
                return f"No per diem rate found for location: {location}"
            elif query_type == "total_expenses":
                project = kwargs.get("project", "")
                cursor.execute("SELECT SUM(amount) FROM expenses WHERE project_name LIKE ?", (f"%{project}%",))
                row = cursor.fetchone()
                conn.close()
                total = row[0] if row and row[0] else 0
                return f"Total expenses for '{project}': ${total:,.2f}"
            else:
                conn.close()
                return f"Unknown query_type: {query_type}. Use approval_limit, per_diem, or total_expenses"
        except Exception as e:
            logger.error(f"Expense query error: {e}")
            return f"Error: {str(e)}"


class Agent:
    def __init__(self, db_path: str, documents_path: str = None, api_key: str = None):
        self.db_path = db_path
        self.api_key = api_key or GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set.")
        self.client = genai.Client(api_key=self.api_key)
        if documents_path is None:
            documents_path = os.path.join(os.path.dirname(db_path), "documents.json")
        self.tools = {
            "employee_lookup": EmployeeLookupTool(db_path),
            "policy_search": PolicySearchTool(documents_path),
            "expense_query": ExpenseQueryTool(db_path),
        }
        self.token_count = 0
        self.total_cost = 0.0
        self.queries_run = 0

        # ── Week 6 guardrails ──────────────────────────────────────────────
        policy_path = ACCESS_POLICY_PATH
        if os.path.exists(policy_path):
            self.access_controller = AccessController(policy_path)
            logger.info("AccessController loaded")
        else:
            self.access_controller = None
            logger.warning(f"access_control.json not found at {policy_path} — access control disabled")

        self.rate_limiter = RateLimiter(max_queries_per_minute=30)
        self.cost_enforcer = CostEnforcer()
        logger.info("Agent initialized with 3 tools and Week 6 guardrails")

    def _build_system_prompt(self, user_role: str) -> str:
        return f"""You are TechCorp's enterprise knowledge assistant. Answer questions using the available tools.

User role: {user_role}

Available tools:
- employee_lookup: Find employee info by name or ID. Use when asked about specific employees.
- policy_search: Search HR, Finance, Engineering, and Compliance policy documents by keyword.
- expense_query: Query expense approval limits (query_type=approval_limit, role=...) or per diem rates (query_type=per_diem, location=...) or project totals (query_type=total_expenses, project=...).

To use a tool, write exactly:
TOOL: tool_name
PARAMS: {{"param1": "value1", "param2": "value2"}}

Rules:
- Use tools to find real data before answering
- If a user asks for salary data and their role is not hr or admin, say access is denied
- If no data is found, say so clearly
- After getting tool results, provide a clear, concise answer
- Do not make up information not in tool results"""

    def _parse_tool_call(self, text: str):
        lines = text.strip().split("\n")
        tool_name = None
        params = {}
        for i, line in enumerate(lines):
            if line.startswith("TOOL:"):
                tool_name = line.replace("TOOL:", "").strip()
            if line.startswith("PARAMS:"):
                params_str = line.replace("PARAMS:", "").strip()
                try:
                    params = json.loads(params_str)
                except Exception:
                    params = {}
        return tool_name, params

    def _estimate_query_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * 0.075
        output_cost = (output_tokens / 1_000_000) * 0.3
        return input_cost + output_cost

    def query(self, user_query: str, user_role: str = "engineer", user_id: str = "anonymous") -> Dict[str, Any]:
        logger.info(f"Processing query from user={user_id} role={user_role}: {user_query}")

        # ── Guardrail 1: Rate limiting ─────────────────────────────────────
        if not self.rate_limiter.is_allowed(user_id):
            remaining = self.rate_limiter.get_remaining_queries(user_id)
            logger.warning(f"Rate limit exceeded for user={user_id}")
            return {
                "answer": "Rate limit exceeded. Please wait before making another query.",
                "tokens_used": 0,
                "cost": 0.0,
                "role": user_role,
                "user_id": user_id,
                "blocked_by": "rate_limiter",
                "remaining_queries": remaining
            }

        # ── Guardrail 2: Budget check ──────────────────────────────────────
        estimated_cost = 0.01
        if not self.cost_enforcer.can_afford_query(user_id, estimated_cost):
            remaining_budget = self.cost_enforcer.get_budget_remaining(user_id)
            logger.warning(f"Budget exceeded for user={user_id}, remaining=${remaining_budget:.4f}")
            return {
                "answer": "Budget limit exceeded for this month. Contact your administrator.",
                "tokens_used": 0,
                "cost": 0.0,
                "role": user_role,
                "user_id": user_id,
                "blocked_by": "cost_enforcer",
                "budget_remaining": remaining_budget
            }

        # ── Log access attempt ─────────────────────────────────────────────
        if self.access_controller:
            self.access_controller.log_access(
                role=user_role,
                resource=f"query:{user_query[:50]}",
                allowed=True,
                user_id=user_id
            )

        system_prompt = self._build_system_prompt(user_role)
        messages = [
            f"System: {system_prompt}",
            f"User: {user_query}"
        ]
        input_tokens = 0
        output_tokens = 0
        tool_results = []
        max_steps = 3

        try:
            for step in range(max_steps):
                prompt = "\n\n".join(messages)
                if tool_results:
                    prompt += "\n\nTool results so far:\n" + "\n".join(tool_results)
                    prompt += "\n\nNow provide a final answer based on the tool results above."

                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                response_text = response.text if hasattr(response, "text") else str(response)

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    input_tokens += getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    output_tokens += getattr(response.usage_metadata, "candidates_token_count", 0) or 0

                tool_name, params = self._parse_tool_call(response_text)

                if tool_name and tool_name in self.tools:
                    logger.info(f"Calling tool: {tool_name} with params: {params}")
                    result = self.tools[tool_name].execute(**params)
                    tool_results.append(f"[{tool_name}]: {result}")
                    messages.append(f"Assistant (step {step+1}): {response_text}")
                    messages.append(f"Tool result: {result}")
                else:
                    cost = self._estimate_query_cost(input_tokens, output_tokens)
                    self.token_count += input_tokens + output_tokens
                    self.total_cost += cost
                    self.queries_run += 1

                    # ── Guardrail 3: Track cost and redact response ────────
                    self.cost_enforcer.add_cost(user_id, user_role, cost)

                    answer = response_text
                    if self.access_controller:
                        answer = self.access_controller.redact_response(user_role, answer)

                    return {
                        "answer": answer,
                        "tokens_used": input_tokens + output_tokens,
                        "cost": cost,
                        "role": user_role,
                        "user_id": user_id
                    }

            cost = self._estimate_query_cost(input_tokens, output_tokens)
            self.token_count += input_tokens + output_tokens
            self.total_cost += cost
            self.queries_run += 1
            self.cost_enforcer.add_cost(user_id, user_role, cost)

            answer = tool_results[-1] if tool_results else "I was unable to find an answer."
            if self.access_controller:
                answer = self.access_controller.redact_response(user_role, answer)

            return {
                "answer": answer,
                "tokens_used": input_tokens + output_tokens,
                "cost": cost,
                "role": user_role,
                "user_id": user_id
            }

        except Exception as e:
            logger.error(f"Agent query error: {e}")
            self.queries_run += 1
            return {
                "answer": f"I encountered an error processing your request: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "role": user_role,
                "user_id": user_id
            }

    def get_metrics(self) -> Dict[str, Any]:
        avg = self.total_cost / self.queries_run if self.queries_run > 0 else 0.0
        audit_log = self.access_controller.get_audit_log() if self.access_controller else []
        denial_count = self.access_controller.get_denial_count() if self.access_controller else 0
        return {
            "total_queries": self.queries_run,
            "total_tokens": self.token_count,
            "total_cost": self.total_cost,
            "avg_cost_per_query": avg,
            "access_denials": denial_count,
            "audit_log_entries": len(audit_log),
            "spending_summary": self.cost_enforcer.get_spending_summary()
        }


if __name__ == "__main__":
    import sys
    try:
        agent = Agent("week5/data/techcorp.db")
        print("Agent initialized successfully")
        result = agent.query("What is the travel policy?", user_id="test_user")
        print(f"Answer: {result['answer'][:500]}")
        print(f"Tokens: {result['tokens_used']}")
        print(f"Cost: ${result['cost']:.6f}")
        print(f"Metrics: {agent.get_metrics()}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
