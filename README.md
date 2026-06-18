# Loan Approval Multi-Agent Underwriter

This is the boilerplate repository for the **Loan Approval Multi-Agent Underwriter** LangGraph Capstone project. It implements a multi-agent decision support system designed to automate loan underwriting workflows while maintaining human-in-the-loop oversight.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Team Roles and Responsibilities](#team-roles-and-responsibilities)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [Testing](#testing)

## Project Overview

The system uses four specialized agents orchestrated via LangGraph:
1. **KYC / Document Agent**: Extracts and validates applicant details.
2. **Credit Risk Agent**: Computes credit score and debt-to-income (DTI) ratio.
3. **Policy / Eligibility Agent**: Validates the application against embedded policy documents via RAG.
4. **Arbitrator Agent**: Resolves conflicts and determines the final recommendation.
5. **Human-in-the-Loop (Streamlit UI)**: Allows Loan Officers to approve or override recommendations.

## Project Structure

```text
RiskPilot/
├── src/
│   ├── agents/               # Member A, B, C, D (Agent implementations)
│   ├── graph/                # Member D (Orchestration & State definitions)
│   ├── tools/                # Member A, B, C (Agent-specific tools)
│   ├── rag/                  # Member C (Vector Store & RAG pipelines)
│   ├── guardrails/           # Member A (Input & Output validation)
│   └── ui/                   # Member D (Officer Review UI)
├── data/
│   ├── policy_docs/          # Policy documents chunked for RAG
│   └── test_applications.json# Test inputs
├── tests/                    # Testing suite (pytest)
└── config/                   # Configuration settings
```

## Team Roles and Responsibilities

* **Member A**: KYC/Document Agent (`src/agents/kyc_agent.py`) + Document Tools + Input Guardrails.
* **Member B**: Credit Risk Agent (`src/agents/credit_agent.py`) + Scoring Tools + State Schema.
* **Member C**: Policy Agent (`src/agents/policy_agent.py`) + RAG system + Policy Docs.
* **Member D**: Arbitrator Agent (`src/agents/arbitrator_agent.py`) + Graph Orchestration (`src/graph/`) + HITL UI.

## Getting Started

1. **Clone and setup environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   make install
   ```
   Or manually:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and fill in your API keys:
   ```env
   OPENAI_API_KEY=sk-...
   LANGCHAIN_API_KEY=lsv2_...
   LANGCHAIN_PROJECT=loan-underwriter
   ```

## Quick Start

```bash
# Run tests
make test

# Launch officer dashboard
make run-demo
```

## Available Commands (Make)

| Target     | Description                                      |
|------------|--------------------------------------------------|
| `install`  | Install dependencies + pre-commit hooks          |
| `test`     | Run pytest with coverage                         |
| `run-demo` | Launch Streamlit officer dashboard               |
| `lint`     | Check code style (black, isort, flake8)          |
| `format`   | Auto-format code (black, isort)                  |
| `clean`    | Remove cache, coverage, chroma_db artifacts      |

## Running the Project

To run the human-in-the-loop dashboard:
```bash
streamlit run src/ui/officer_dashboard.py
```

## Testing

```bash
pytest -v --cov=src --cov-report=term-missing
```
