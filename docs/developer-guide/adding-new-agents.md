# Adding New Agents & Custom Tools

This guide explains how to add new domain analysts or custom data tools to the TradingAgents LangGraph architecture.

---

## 1. Creating a New Analyst Agent

1. **Define the Node Function** in `tradingagents/agents/analysts/my_custom_analyst.py`:
   ```python
   from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
   from tradingagents.agents.utils.agent_utils import get_language_instruction, build_instrument_context
   
   def create_custom_analyst(llm):
       def custom_analyst_node(state):
           current_date = state["trade_date"]
           ticker = state["company_of_interest"]
           asset_type = state.get("asset_type", "stock")
           instrument_context = build_instrument_context(ticker, asset_type, trade_date=current_date)
           
           tools = [my_custom_tool]
           system_message = (
               "You are a specialized Market Microstructure Analyst..."
               + get_language_instruction()
           )
           
           # Bind prompt and invoke tool-calling loop
           chain = prompt | llm.bind_tools(tools)
           result = chain.invoke(state["custom_messages"])
           
           return {
               "custom_messages": [result],
               "custom_report": result.content if len(result.tool_calls) == 0 else "",
           }
       return custom_analyst_node
   ```

2. **Register the State Channel** in `tradingagents/agents/utils/agent_states.py`:
   ```python
   class AgentState(TypedDict):
       ...
       custom_messages: Annotated[list, add_messages]
       custom_report: str
   ```

3. **Register Graph Node & Fan-Out** in `tradingagents/graph/setup.py`:
   - Add to `analyst_nodes` mapping and include in the fan-in join barrier.

---

## 2. Adding a Custom Data Tool

1. Create the tool function with LangChain `@tool` decorator:
   ```python
   from langchain_core.tools import tool
   
   @tool
   def get_order_flow_imbalance(symbol: str, trade_date: str) -> str:
       """Calculate institutional order flow imbalance as of trade_date."""
       # Perform causal Point-in-Time data lookup
       return f"Order Flow Delta for {symbol}: +15.4M shares net buy"
   ```
2. Bind the tool to the target analyst node in `setup.py`.
