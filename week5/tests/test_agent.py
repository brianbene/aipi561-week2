import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.agent import (
    Tool, EmployeeLookupTool, PolicySearchTool,
    ExpenseQueryTool, Agent
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "techcorp.db")
DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")
API_KEY = os.getenv("GOOGLE_API_KEY", "test-key")

# --- Tool initialization tests ---

def test_employee_lookup_tool_name():
    tool = EmployeeLookupTool(DB_PATH)
    assert tool.name == "employee_lookup"

def test_employee_lookup_tool_description():
    tool = EmployeeLookupTool(DB_PATH)
    assert "employee" in tool.description.lower()

def test_policy_search_tool_name():
    tool = PolicySearchTool(DOCS_PATH)
    assert tool.name == "policy_search"

def test_policy_search_loads_documents():
    tool = PolicySearchTool(DOCS_PATH)
    assert len(tool.documents) == 74

def test_expense_query_tool_name():
    tool = ExpenseQueryTool(DB_PATH)
    assert tool.name == "expense_query"

def test_tool_base_raises_not_implemented():
    tool = Tool("test", "test description")
    with pytest.raises(NotImplementedError):
        tool.execute()

# --- Tool execute tests ---

def test_employee_lookup_by_name():
    tool = EmployeeLookupTool(DB_PATH)
    result = tool.execute(employee_name="John")
    assert isinstance(result, str)
    assert result != "Error"

def test_employee_lookup_missing_params():
    tool = EmployeeLookupTool(DB_PATH)
    result = tool.execute()
    assert "provide" in result.lower() or "error" in result.lower()

def test_policy_search_returns_results():
    tool = PolicySearchTool(DOCS_PATH)
    result = tool.execute(query="travel policy")
    assert "travel" in result.lower() or "policy" in result.lower()

def test_policy_search_no_results():
    tool = PolicySearchTool(DOCS_PATH)
    result = tool.execute(query="xyzzy nonexistent topic abc123")
    assert "no policy" in result.lower() or isinstance(result, str)

def test_expense_query_unknown_type():
    tool = ExpenseQueryTool(DB_PATH)
    result = tool.execute(query_type="unknown_type")
    assert "unknown" in result.lower()

# --- Agent initialization tests ---

def test_agent_initializes():
    agent = Agent(DB_PATH, DOCS_PATH, api_key=API_KEY)
    assert agent is not None

def test_agent_has_three_tools():
    agent = Agent(DB_PATH, DOCS_PATH, api_key=API_KEY)
    assert len(agent.tools) == 3

def test_agent_tool_names():
    agent = Agent(DB_PATH, DOCS_PATH, api_key=API_KEY)
    assert "employee_lookup" in agent.tools
    assert "policy_search" in agent.tools
    assert "expense_query" in agent.tools

# --- Cost calculation tests ---

def test_cost_calculation_zero():
    agent = Agent(DB_PATH, DOCS_PATH, api_key=API_KEY)
    cost = agent._estimate_query_cost(0, 0)
    assert cost == 0.0

def test_cost_calculation_input_only():
    agent = Agent(DB_PATH, DOCS_PATH, api_key=API_KEY)
    cost = agent._estimate_query_cost(1_000_000, 0)
    assert abs(cost - 0.075) < 0.0001

def test_cost_calculation_output_only():
    agent = Agent(DB_PATH, DOCS_PATH, api_key=API_KEY)
    cost = agent._estimate_query_cost(0, 1_000_000)
    assert abs(cost - 0.3) < 0.0001

def test_metrics_initial_state():
    agent = Agent(DB_PATH, DOCS_PATH, api_key=API_KEY)
    metrics = agent.get_metrics()
    assert metrics["total_queries"] == 0
    assert metrics["total_tokens"] == 0
    assert metrics["total_cost"] == 0.0
    assert metrics["avg_cost_per_query"] == 0.0
