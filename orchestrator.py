"""Orchestrator: routes user requests to sub-agents, each wrapped as a tool."""
from agent_core import build_agent, chat_loop
from shoe_agent import shoe_tool

NAME = "orchestrator"

SYSTEM = """You are the orchestrator. You do not act directly; you delegate to specialist agents (your tools).
- Pick the right agent(s) for each part of the request. Call several if the request spans domains.
- Sub-agents cannot see this conversation. Always pass a complete, self-contained request
  (names, IDs, amounts, what to do) in the `request` argument.
- If a sub-agent asks for missing info, ask the user, then call it again.
- Combine the results into one short answer. Never invent results.
- If no agent fits, answer briefly yourself and say which capability is missing.
- Never answer a domain question yourself. All shoe questions, including reviews and
  general recommendations, go to shoe_store_agent."""

def build_sub_agents():
    """Register sub-agents here. Each one is just a tool to the orchestrator."""
    return [
        shoe_tool(),
        # github_tool(),   # next: build github_agent.py the same way as shoe_agent.py
        # linear_tool(),
        # travel_tool(),
    ]


def build_orchestrator(sub_agents=None, llm=None):
    # Sub-agents ask for approval themselves, so the orchestrator has no sensitive tools
    return build_agent(NAME, SYSTEM, sub_agents or build_sub_agents(), llm=llm)


if __name__ == "__main__":
    chat_loop(
        build_orchestrator(),
        "Orchestrator ready. Try: 'Jane Doe needs running shoes - order the cheapest in stock and email her'.",
    )
