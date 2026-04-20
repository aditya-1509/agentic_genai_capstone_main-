# Deep Dive: Inside the Agentic Workflow

This document provides an under-the-hood explanation of exactly how your Agentic AI operates, manages memory, uses tools, and integrates with the rest of your system (ML, RAG, and Web Search). This is the core logic you need to explain to an examiner who asks, *"How does the agent actually work internally?"*

---

## 1. What is an Agentic Workflow?
A traditional LLM chatbot is **linear** (User inputs text -> LLM outputs text). 
An **Agentic Workflow** is **cyclic and autonomous**. The LLM is given a Persona (System Prompt), Memory (State), and "Hands" (Tools). Instead of just answering, it enters a loop where it can *reason* about a problem, *decide* to execute Python code (tools), analyze the result of that code, and then synthesize a final response.

In your project, this is managed by **LangGraph**, a framework designed to orchestrate these cyclic execution graphs.

---

## 2. State Management (The Agent's Memory)
If an agent loops multiple times, it must remember what it did in the previous loop. In LangGraph, this is handled by **State**.

### Internal Mechanics:
*   **`AgentState` TypedDict**: Your project defines a schema containing a `messages` array. This is the global memory of the graph.
*   **The `add_messages` Reducer**: In a standard Python dictionary, if you update `state["messages"]`, it overwrites the old value. The `add_messages` reducer changes this behavior so it **appends** new messages instead.
*   **Context Preservation**: As the agent loops, the state grows:
    1.  `HumanMessage` (User query)
    2.  `AIMessage` (LLM decides to call a tool)
    3.  `ToolMessage` (The output of the RAG search)
    4.  `AIMessage` (The final answer)
*   Because nothing is overwritten, the LLM has complete historical context. It knows *what* the user asked, *why* it searched the database, and *what* the database returned.

---

## 3. Tool Binding (Giving the Agent Hands)
An LLM cannot directly search a database or run Python code. It can only generate text. **Tool Binding** bridges this gap.

### Internal Mechanics:
1.  **The `@tool` Decorator**: In `graph.py`, you wrap both `search_regulations_tool` and `search_web_tool` with this Langchain decorator.
2.  **JSON Schema Generation**: Behind the scenes, the decorator parses the Python functions' names, argument types, and **Docstrings**. It compiles these into structured JSON Schemas.
3.  **API Payload Injection**: When LangGraph sends a request to the Groq API (Llama-3), it attaches an array of these JSON Schemas in a special `tools` parameter. 
4.  **The Attention Mechanism**: The LLM reads the system prompt ("ALWAYS try `search_regulations_tool` first...") and the tool docstrings. If the user asks a regulatory question, the LLM's neural network recognizes the semantic match, prioritizing the local RAG tool. If that tool fails, it knows the `search_web_tool` exists as a fallback.

---

## 4. The ReAct Architecture (Reasoning and Acting)
Your agent is built using the `create_react_agent` function. This forces the LLM into a specific Thought-Action-Observation loop, which is now significantly more powerful with multiple tools.

### Internal Execution Sequence (Multi-Tool Fallback Example):
When the user asks: *"What is the current Repo Rate?"*

1.  **Reasoning (Thought 1)**: LangGraph sends the `AgentState` to the Groq API. The LLM thinks: *"I need to find the Repo Rate. The system prompt tells me to use `search_regulations_tool` first."*
2.  **Acting (Action 1)**: The LLM halts standard text generation. It outputs a structured JSON block requesting the `search_regulations_tool`.
3.  **Observation 1**: LangGraph intercepts the JSON, runs ChromaDB locally, and gets back: *"No regulatory guidelines found."* It appends this failure to the `AgentState`.
4.  **Reasoning (Thought 2)**: LangGraph sends the updated state back to Groq. The LLM reads its own failure and thinks: *"The local database didn't have it. I need to use my fallback tool, `search_web_tool`."*
5.  **Acting (Action 2)**: The LLM outputs a *second* JSON block, this time requesting `search_web_tool`.
6.  **Observation 2**: LangGraph executes the DuckDuckGo search and appends the live internet results to the `AgentState`.
7.  **Synthesis**: The 70B LLM reads the internet results and types out a natural language response for the user.
8.  **Termination**: The LLM generates plain text instead of a tool call, routing to the internal `__end__` node.

---

## 5. How is it connected to RAG?
RAG (Retrieval-Augmented Generation) is simply the internal logic of the Agent's Primary Tool.

*   The Agent doesn't know what ChromaDB, embeddings, or MMR are. 
*   To the Agent, the `search_regulations_tool` is just a black box that takes a string and returns a string.
*   The actual RAG logic (vectorizing the query via `all-MiniLM-L6-v2`, doing an MMR semantic search in ChromaDB, and formatting the chunks) happens entirely within `retriever.py`.
*   **The Connection**: The Agent acts as the *orchestrator* that decides *which* tool to use, while the RAG pipeline is the *engine* that executes the specific local vector search.

---

## 6. How is it connected to Machine Learning?
The Agent itself **does not do any math or predict risk**. LLMs are notoriously bad at precise mathematics and tabular machine learning. 

### The Connection (Contextual Grounding):
1.  **Execution Order**: When the user clicks submit, the pure Python XGBoost pipeline (`run_ml_pipeline`) runs *first*. It calculates the probability (e.g., 0.65), applies the hardcoded Hybrid Guardrail (Review), and ranks the risk drivers.
2.  **State Injection**: Before the Agentic Chat is even allowed to start, the output of the ML pipeline is injected as a hidden **`SystemMessage`** at the very beginning of the `AgentState`.
3.  **Grounding**: When the user chats with the Agent, the Agent reads this hidden message. It treats the ML Output as an irrefutable fact. 
4.  **The Result**: If the user asks *"Why was I rejected?"*, the Agent looks at the injected ML Feature Contributions (e.g., *Loan-to-Income is 45%*). It synthesizes this into natural language: *"You were classified as high risk primarily because your Loan-to-Income ratio is 45%, which is too high."*

This architectural separation ensures the system relies on deterministic XGBoost math for the actual decision, while utilizing the LLM purely for explainability, dynamic tool usage, and natural language communication.
