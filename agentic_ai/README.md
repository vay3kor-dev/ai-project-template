# Agentic AI Chatbot using LangGraph

## Overview

This project implements an Agentic AI chatbot using LangGraph.

The agent automatically analyses user input, selects the most appropriate tool, executes the tool, and generates a final answer.

---

# Architecture

User Question
↓
Router
↓
Knowledge Tool
OR
Calculator Tool
OR
Date Tool
↓
Final Answer Node
↓
Response

---

# Features

## Knowledge Base Tool

Uses ChromaDB for semantic search.

Example:

```text
What is RAG?
```

---

## Calculator Tool

Performs mathematical calculations.

Example:

```text
25 * 20
```

---

## Date Tool

Handles date-related queries.

Examples:

```text
Today's date
Yesterday's date
Tomorrow's date
```

---

# Folder Structure

```text
agentic_ai/
│
├── agent_chatbot.py
├── tools.py
├── build_db.py
├── knowledge.txt
├── chroma_db/
```

---

# Prerequisites

## Activate Environment

```powershell
.venv\Scripts\activate
```

---

## Install Dependencies

```powershell
pip install -r requirements-agentic.txt
```

---

## Verify Ollama

Required models:

```powershell
ollama list
```

Expected:

```text
qwen3
nomic-embed-text
```

Pull if missing:

```powershell
ollama pull qwen3
ollama pull nomic-embed-text
```

---

# Step 1 - Build Knowledge Base

Update:

```text
knowledge.txt
```

Then build:

```powershell
python agentic_ai/build_db.py
```

Expected:

```text
Database created successfully.
```

---

# Step 2 - Start Agent

Run:

```powershell
python agentic_ai/agent_chatbot.py
```

Expected:

```text
===== Agentic AI Terminal Chatbot =====
```

---

# Test Scenarios

## Knowledge Tool

```text
What is RAG?
What is ChromaDB?
What is LangGraph?
```

Expected:

```text
Selected Tool:
Knowledge Base Tool
```

---

## Calculator Tool

```text
2+3

25*8

100/5
```

Expected:

```text*Selected Tool:
*alculator Tool
```

*--

## Date Tool

```text*What is today's date?

Yesterday*s date?

Tomorrow's date?
```

*xpected*

```text
Selected Tool:
Date Tool*```

*--

# Core Concepts Implemented

-*Agentic AI
- LangGraph
- Shared St*te
- Routing*Logic
- Tool Calling
- Semantic Re*rieval
- ChromaDB
- Embeddings
- W*rkflow Orchestration

---

* Troubles*ooting

## ChromaDB Missing

Re*uild:

```powers*ell
python agentic_ai/build_db*py
```

*--

## Ollama Connection Error

Ve*ify:

```powershell
ollama list
``*

---

## Model Missing

Pull*again:

```powers*ell
ollama pull qwen3
ollama*pull nomic-embed*text
```

---

# Expected Outcome
*The chatbot*should automatically:

1**Understand user intent
2.*Select the correct tool
3. Execute*the tool
4. Generate*the answer*
without requiring*users to specify which tool should*be used.
````*
