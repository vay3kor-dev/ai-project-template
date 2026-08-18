# Enterprise AI Learning Portfolio

## Overview

This repository documents my AI and Generative AI learning journey, covering the progression from basic chatbot development to advanced Agentic AI systems.

The objective of this learning program was to gain hands-on experience with:

- Python-based AI Development
- Local Small Language Models (SLMs)
- Enterprise Large Language Models (LLMs)
- Retrieval Augmented Generation (RAG)
- Embedding Models
- Vector Databases
- Multimodal AI
- Voice AI
- LangChain
- LangGraph
- Agentic AI
- Bosch Model Farm (BMF)

The projects demonstrate how modern AI systems are designed, built, tested, and deployed using both local and enterprise AI platforms.

---

# Learning Journey

## Phase 1: AI Development Environment Setup

### Topics Covered

- Python
- Virtual Environments
- VS Code
- Dependency Management
- Git
- GitHub

### Purpose

To establish a complete development environment capable of building and running AI applications locally and within enterprise environments.

---

## Phase 2: Local AI Models with Ollama

### Technologies

- Ollama
- Phi-3
- Qwen3

### Purpose

To understand how Large Language Models and Small Language Models can be executed locally without relying on cloud services.

### Projects

- chatbot.py
- phi3_chatbot.py
- test_ollama.py
- test_phi3.py

### Key Learnings

- Local Model Execution
- Prompt Engineering
- Model Inference
- AI Application Development

---

## Phase 3: Enterprise AI with Bosch Model Farm

### Technologies

- Bosch Model Farm (BMF)
- Enterprise APIs

### Purpose

To understand how enterprise-grade AI systems provide secure access to approved Large Language Models.

### Projects

- chatbot_bmf.py
- test_bmf.py

### Key Learnings

- Enterprise AI Access
- Authentication
- API Integration
- Secure AI Consumption

---

## Phase 4: Retrieval Augmented Generation (RAG)

### Technologies

- ChromaDB
- nomic-embed-text
- RAG Architecture

### Purpose

To improve response accuracy by retrieving relevant information before generating answers.

### Projects

- text_rag_chatbot.py
- phi3_rag_chatbot.py

### Workflow

```text
Question
    ↓
Retriever
    ↓
Knowledge Base
    ↓
LLM
    ↓
Answer
```

### Key Learnings

- Retrieval Augmented Generation
- Knowledge Grounding
- Hallucination Reduction
- Context-Aware Responses

---

## Phase 5: Embeddings and Vector Databases

### Technologies

- nomic-embed-text
- ChromaDB

### Purpose

To enable semantic understanding and semantic search.

### Projects

- build_db.py
- knowledge.txt

### Key Learnings

- Text Chunking
- Embeddings
- Vector Search
- Semantic Similarity
- Vector Database Architecture

---

## Phase 6: Multimodal AI

### Technologies

- Vision Language Models
- Image Processing

### Projects

- multimodal_chatbot.py
- image_to_text.py
- image_test.py

### Purpose

To understand how AI systems process multiple input types.

### Key Learnings

- Image Understanding
- Visual Question Answering
- Multimodal Workflows
- AI-driven Image Interpretation

---

## Phase 7: Voice AI

### Projects

- voice_chatbot.py
- voice_test.py

### Purpose

To build voice-enabled AI applications.

### Workflow

```text
Voice
    ↓
Speech-to-Text
    ↓
Language Model
    ↓
Response
```

### Key Learnings

- Audio Processing
- Speech Interfaces
- Conversational AI
- Voice-Based Applications

---

## Phase 8: Multimodal RAG

### Projects

- multimodal_rag.py

### Purpose

To combine image understanding with retrieval systems.

### Key Learnings

- Image-Based Retrieval
- Context-Aware Question Answering
- Multimodal Retrieval Systems

---

## Phase 9: Cascaded AI Architecture

### Technologies

- Phi-3
- Bosch Model Farm

### Project

- cascadechat.py

### Purpose

To optimize performance and cost using multiple AI models.

### Workflow

```text
Question
    ↓
Phi-3
    ↓
Confidence Check
    ↓
Bosch Model Farm
```

### Key Learnings

- Model Orchestration
- Hybrid AI Systems
- Cost Optimization
- Cascaded AI Architectures

---

## Phase 10: Agentic AI

### Technologies

- LangGraph
- ChromaDB
- Ollama
- Qwen3

### Folder

```text
agentic_ai/
```

### Projects

- agent_chatbot.py
- tools.py
- build_db.py
- knowledge.txt

### Purpose

To build an AI system capable of reasoning, selecting tools, executing actions, and generating responses autonomously.

---

## Implemented Tools

### Knowledge Base Tool

Retrieves relevant information from ChromaDB and provides context for answer generation.

### Calculator Tool

Performs arithmetic and mathematical calculations.

### Date Tool

Handles current, past, and future date-related questions.

---

## Agent Workflow

```text
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
```


# Repository Structure

```text
agentic_ai/
│
├── agent_chatbot.py
├── build_db.py
├── tools.py
├── knowledge.txt
│
src/
│
├── chatbot.py
├── phi3_chatbot.py
├── chatbot_bmf.py
├── text_rag_chatbot.py
├── phi3_rag_chatbot.py
├── multimodal_chatbot.py
├── multimodal_rag.py
├── cascadechat.py
├── voice_chatbot.py
│
docs/
│
└── sample.pdf
```

---

# Key Outcome

Through these projects, I gained practical experience in:

- AI Application Development
- Enterprise AI Systems
- Retrieval Augmented Generation (RAG)
- Vector Databases
- Embedding Models
- Multimodal AI
- Voice AI
- Agentic AI
- Workflow Orchestration
- Enterprise AI Architectures

This repository demonstrates the complete progression from simple chatbot development to enterprise-style Agentic AI systems capable of retrieval, reasoning, tool execution, and autonomous decision making.
