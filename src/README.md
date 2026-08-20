# AI Chatbots, RAG, Multimodal and Cascade AI Projects

## Overview

This folder contains all chatbot, RAG, multimodal, voice AI, and cascade architecture projects developed during the AI learning journey.

These projects demonstrate:

- Local AI models using Ollama
- Enterprise AI using Bosch Model Farm
- RAG implementations
- Vector databases
- Multimodal AI
- Voice AI
- Cascaded AI architectures

---

# Prerequisites

## Python

Install Python 3.10 or later.

Verify installation:

```powershell
python --version
```

---

## Create Virtual Environment

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\activate
```

---

## Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## Start Ollama

Verify:

```powershell
ollama list
```

Required models:

```powershell
phi3
qwen3
nomic-embed-text
```

Pull if missing:

```powershell
ollama pull phi3
ollama pull qwen3
ollama pull nomic-embed-text
```

---

# Project Guide

## Basic Chatbot

File:

```text
chatbot.py
```

Run:

```powershell
python src/chatbot.py
```

Purpose:

Simple AI chatbot.

---

## Phi-3 Chatbot

File:

```text
phi3_chatbot.py
```

Run:

```powershell
python src/phi3_chatbot.py
```

Purpose:

Local chatbot using Microsoft Phi-3.

---

## Bosch Model Farm Chatbot

File:

```text
chatbot_bmf.py
```

Requirements:

- Bosch Model Farm account
- API key
- Configured .env

Run:

```powershell
python src/chatbot_bmf.py
```

Purpose:

Enterprise chatbot using Bosch Model Farm.

---

## Text RAG Chatbot

Files:

```text
text_rag_chatbot.py
build_db.py
data/knowledge.txt
```

Build Database:

```powershell
python src/build_db.py
```

Run:

```powershell
python src/text_rag_chatbot.py
```

Purpose:

Question answering from local knowledge.

---

## Phi-3 RAG Chatbot

Run:

```powershell
python src/phi3_rag_chatbot.py
```

Purpose:

RAG with Phi-3 as answer generation model.

---

## Multimodal Chatbot

Run:

```powershell
python src/multimodal_chatbot.py
```

Purpose:

Accepts image and text inputs.

---

## Multimodal RAG

Run:

```powershell
python src/multimodal_rag.py
```

Purpose:

Image understanding combined with retrieval.

---

## Voice Chatbot

Run:

```powershell
python src/voice_chatbot.py
```

Purpose:

Voice-enabled AI assistant.

---

## Cascade AI Chatbot

Run:

```powershell
python src/cascadechat.py
```

Purpose:

Uses Phi-3 first and escalates difficult questions to Bosch Model Farm.

Workflow:

Question
↓
Phi-3
↓
Confidence Check
↓
Bosch Model Farm
↓
Response

---

# Technologies Used

- Python
- Ollama
- Phi-3
- Qwen3
- Bosch Model Farm
- ChromaDB
- nomic-embed-text
- RAG
- Multimodal AI

---

# Expected Outcome

Users can test:

- Chatbots
- RAG Systems
- Multimodal AI
- Voice AI
- Cascade Architectures

using the commands documented above.
