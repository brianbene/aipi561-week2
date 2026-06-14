# TechCorp Knowledge Agent — Architecture Document

## System Overview

The TechCorp Knowledge Agent is an AI-powered question-answering system that combines
a large language model (Gemini 2.5 Flash) with structured data tools to answer business
questions about TechCorp employees, policies, and expenses.

## Query Flow

1. User submits a question via the chat UI or POST /agent/query
2. The Agent builds a system prompt describing available tools and user role
3. Gemini LLM receives the prompt and decides which tool to call
4. The Agent parses the LLM response, extracts tool name and parameters
5. The tool executes (database query or document search) and returns results
6. Results are passed back to the LLM for synthesis into a final answer
7. The answer is returned to the user with token count and cost

## Tool Definitions

### employee_lookup
- Purpose: Find employee records by name or ID
- Data source: SQLite employees table (10,000 records)
- Parameters: employee_name (partial match) or employee_id (exact match)
- Returns: JSON with id, name, department, role
- Access: All roles (salary fields excluded for non-HR roles)

### policy_search
- Purpose: Search 74 TechCorp policy documents by keyword
- Data source: documents.json loaded at startup
- Parameters: query (keyword string), limit (max results, default 3)
- Retrieval method: Keyword scoring — counts matching words across title, content, category
- Returns: JSON array with document id, title, category, and 500-char snippet

### expense_query
- Purpose: Query expense approval limits, per diem rates, project totals
- Data source: SQLite expense_policies and expenses tables
- Parameters: query_type (approval_limit / per_diem / total_expenses) + role/location/project
- Returns: Formatted string with the requested value

## Reasoning Loop

The agent uses a text-based reasoning loop (max 3 steps):
- Step 1: LLM receives system prompt + user question, decides which tool to call
- Step 2: Tool executes, result is appended to conversation context
- Step 3: LLM synthesizes a final answer from tool results
- If no tool call is detected, the LLM answer is returned directly
- If max steps reached, the last tool result is returned as the answer

## Cost Tracking

Gemini 2.5 Flash pricing:
- Input tokens: $0.075 per 1M tokens
- Output tokens: $0.300 per 1M tokens

Observed costs from 10 sample queries:
- Average cost per query: $0.000107
- Total for 10 queries: ~$0.000966
- Projected cost at 1,000 queries/day: $0.107/day = $39/year
- Projected cost at 10,000 queries/day: $1.07/day = $390/year

## Fallback Behavior

| Situation | Fallback |
|---|---|
| Tool not found in response | Return LLM answer directly |
| Tool execution error | Return error message, do not crash |
| API rate limit (429) | Catch exception, return graceful error message |
| No documents match query | Return "No policy documents found" message |
| Employee not found | Return "Employee not found" message |
| Max reasoning steps reached | Return last tool result |

## Access Control

The system prompt includes the user role (engineer, manager, hr, admin).
The LLM is instructed to deny salary data requests for non-HR/admin roles.
Role is passed through to the response for audit logging.
Future enhancement: enforce role checks at the tool level before execution.

## Design Decisions

### Why keyword search instead of vector embeddings?
Keyword search requires no embedding API calls, has zero additional cost, loads
instantly at startup, and works well for structured policy documents with consistent
terminology. Vector embeddings would improve recall for semantic queries but add
latency and cost per query.

### Why text-based tool calling instead of native function calling?
The text-based TOOL:/PARAMS: pattern works across model versions without requiring
specific SDK support. Native function calling would be more reliable but ties the
implementation to a specific API version.

### Why SQLite instead of a cloud database?
SQLite ships with Python, requires no infrastructure, and the 49MB database loads
in milliseconds. For production at scale, a managed PostgreSQL instance would be
appropriate.

### Why Gemini 2.5 Flash instead of Pro?
Flash is significantly cheaper and faster. For policy Q&A with retrieved context,
Flash provides sufficient quality. Pro would be appropriate for complex multi-step
reasoning tasks.

## Limitations

- Policy search is keyword-based; semantic queries may miss relevant documents
- Daily free tier quota limits testing (20 requests/day on free tier)
- No caching — identical queries make full LLM calls each time
- No persistent conversation history between sessions
- Role enforcement is prompt-based, not enforced at the tool level
