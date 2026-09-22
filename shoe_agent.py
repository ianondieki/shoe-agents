"""Shoe store sub-agent. Run alone, or import shoe_tool() into the orchestrator."""
from agent_core import agent_as_tool, build_agent, chat_loop, cli_approve
from db import init_db
from tools import SENSITIVE, TOOLS

NAME = "shoe_store_agent"

SYSTEM = """You are the shoe store agent.
- Look up the customer first to get their CustomerID, email, preferred activity and shoe size.
- Recommend in-stock shoes that match their activity. Never order a shoe with InvCount 0.
- Answer from the database. Only call search_web when the customer explicitly asks you to look
  something up online - the words search, review, compare, online or news. A recommendation, a
  price, a size or a stock question is answered from the database alone, never from the web.
- If and only if you searched, cite the URLs you used.
- After placing or cancelling an order, email a confirmation if asked.
- If the request lacks the customer name, say so instead of guessing.
- Reply with a short, factual summary (include IDs) - another agent may read it."""

DESCRIPTION = (
    "Shoe store specialist. Handles customer lookup, shoe recommendations and stock, "
    "placing/cancelling/deleting orders, order history, shoe-related web search and "
    "confirmation emails. Pass a complete request including the customer's name, "
    "e.g. 'Jane Doe wants a running shoe under $100, place the order and email her'."
)


def build_shoe_agent(approve=cli_approve, llm=None):
    init_db()
    return build_agent(NAME, SYSTEM, TOOLS, SENSITIVE, approve, llm)


def shoe_tool(approve=cli_approve, llm=None):
    return agent_as_tool(build_shoe_agent(approve, llm), NAME, DESCRIPTION)


if __name__ == "__main__":
    chat_loop(build_shoe_agent(), "Shoe agent ready. Try: 'I'm Jane Doe, recommend a running shoe'.")
