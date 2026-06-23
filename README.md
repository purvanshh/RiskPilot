# RiskPilot — Loan Approval Multi-Agent Underwriter

> **Status: ✅ Complete — All PRD requirements met · 123 tests passing · 6/6 demo cases verified**

A production-ready, multi-agent AI system that automates loan underwriting while preserving mandatory human oversight. Built with **LangGraph**, **ChromaDB**, and **LangSmith** as a capstone project.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Policy Documents](#policy-documents)
- [Quick Start](#quick-start)
- [Running the System](#running-the-system)
- [Testing](#testing)
- [Observability](#observability)
- [Team & Phase Allocation](#team--phase-allocation)
- [PRD Compliance Checklist](#prd-compliance-checklist)

---

## Overview

RiskPilot processes loan applications through four specialised AI agents orchestrated by a LangGraph state machine. Every recommendation is surfaced to a human loan officer for final sign-off before any decision reaches the applicant.

**Core value proposition:**
- 🚀 **Speed** — Automates multi-step underwriting in seconds
- 🎯 **Accuracy** — Consistent policy enforcement via RAG over 6 policy documents
- 🔒 **Safety** — Mandatory human-in-the-loop; no autonomous approvals or denials
- 📋 **Auditability** — Every decision event logged to `logs/audit.jsonl`
- 🔭 **Observability** — Full LangSmith trace per application run

---

## Architecture

```
Loan Application
      │
      ▼
┌─────────────┐
│ Input        │  Pydantic schema validation, doc count/type/size checks
│ Guardrails   │  → structured refusal if violations found
└──────┬───────┘
       │
       ▼
┌─────────────┐
│  KYC Agent  │  Document parsing (PyPDF2 / Pillow), field extraction
│             │  Identity & income verification, fraud detection
└──────┬───────┘
       │ fraud / missing docs → Human Review
       │ clean ↓
┌─────────────┐
│ Credit Risk │  PRD scoring formula, DTI, default probability
│    Agent    │  Risk categorisation (low / medium / high / very_high)
└──────┬───────┘
       │
       ▼
┌─────────────┐
│   Policy    │  RAG retrieval over 6 policy docs (ChromaDB + BGE embeddings)
│    Agent    │  LTV, DTI hard-cap, credit floor, employment tenure checks
└──────┬───────┘
       │
       ▼
┌─────────────┐
│ Arbitrator  │  Conflict resolution, weighted voting, confidence scoring
│    Agent    │  → approve / deny / review_required
└──────┬───────┘
       │ always
       ▼
┌─────────────┐
│ Human-in-   │  Loan officer reviews recommendation via Streamlit UI
│  the-Loop   │  Can approve, deny, or override with documented reason
└─────────────┘
```

All nodes share a single typed `LoanApplicationState` (Pydantic) and are protected by `@graceful_fallback` and `@timeout_resilience(30s)` decorators.

---

## Project Structure

```text
RiskPilot/
├── src/
│   ├── agents/
│   │   ├── kyc_agent.py           # KYC / Document agent
│   │   ├── credit_agent.py        # Credit Risk agent
│   │   ├── policy_agent.py        # Policy / Eligibility agent (RAG)
│   │   └── arbitrator_agent.py    # Arbitrator + conflict resolution
│   ├── graph/
│   │   ├── state.py               # LoanApplicationState schema + decorators
│   │   ├── graph.py               # LangGraph graph definition
│   │   └── edges.py               # Conditional routing logic
│   ├── tools/
│   │   ├── document_tools.py      # PDF/OCR parsing, field extraction (LLM + regex)
│   │   ├── credit_tools.py        # Scoring formula, DTI, risk classification
│   │   ├── policy_tools.py        # LTV calculator, RAG-backed policy retriever
│   │   ├── data_loader.py         # Loads test_applications.json → LoanApplicationState
│   │   └── human_review_tool.py   # Programmatic HITL decision API
│   ├── rag/
│   │   ├── embeddings.py          # BGE sentence-transformer embeddings + MockEmbeddings
│   │   ├── policy_loader.py       # Markdown chunker + ChromaDB indexer
│   │   ├── vector_store.py        # ChromaDB client wrapper
│   │   ├── retriever.py           # PolicyRetriever with similarity scoring
│   │   ├── splitter.py            # Section-aware markdown splitter
│   │   ├── evaluator.py           # RAG quality evaluator
│   │   └── cache.py               # Indexing freshness check
│   ├── guardrails/
│   │   ├── input_validation.py    # Pydantic schema + doc count/type/size checks
│   │   ├── output_validation.py   # Post-decision confidence & schema guards
│   │   └── audit_logger.py        # Append-only JSONL audit trail
│   ├── ui/
│   │   └── officer_dashboard.py   # Streamlit loan officer review UI
│   └── main.py                    # CLI entry point with LangSmith tracing
├── data/
│   ├── policy_docs/               # 6 policy documents (RAG source)
│   ├── synthetic_docs/            # 20 synthetic PDFs for APP-001..APP-006
│   └── test_applications.json     # 6 structured test cases
├── tests/                         # 123 tests across 14 test files
├── docs/
│   ├── graph.mmd                  # LangGraph Mermaid diagram source
│   └── graph.png                  # Rendered graph visualisation
├── scratch/
│   ├── run_demo_cases.py          # End-to-end demo runner (all 6 cases)
│   └── generate_graph_viz.py      # Graph visualisation generator
├── config/
│   └── settings.yaml              # Thresholds and model configuration
├── logs/
│   └── audit.jsonl                # Append-only audit log (auto-generated)
├── Makefile                       # Development commands
├── requirements.txt
├── .env.example                   # Environment variable template
└── .pre-commit-config.yaml        # black + isort + flake8 hooks
```

---

## Policy Documents

The RAG pipeline indexes **6 policy documents** from `data/policy_docs/`:

| File | Content |
|------|---------|
| `credit_policy.md` | Credit score tiers (Low ≥720 / Medium 650–719 / High 600–649 / Very High <600) |
| `dti_policy.md` | Debt-to-Income thresholds; hard cap at 50% DTI |
| `employment_policy.md` | Minimum 12 months employment; stability requirements |
| `income_verification.md` | Document requirements; payslip vs bank-statement reconciliation |
| `ltv_policy.md` | Loan-to-Value limits; 80% standard, 85% hard cap |
| `general_guidelines.md` | Override authority, exception handling, audit trail, regulatory compliance (ECOA/FHA) |

---

## Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/purvanshh/RiskPilot.git
cd RiskPilot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
make install
# or: pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-...              # Optional: enables LLM-based field extraction
LANGSMITH_TRACING=true             # Enable LangSmith observability
LANGSMITH_API_KEY=lsv2_...        # Your LangSmith API key
LANGSMITH_PROJECT=RiskPilot
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

> **Note:** The system works without `OPENAI_API_KEY` — it falls back to a robust regex-based field extractor. Without `LANGSMITH_API_KEY`, traces are not captured but the pipeline runs normally.

---

## Running the System

### Run a single application (with LangSmith trace)

```bash
python -m src.main                  # runs APP-001 (Alice Johnson)
python -m src.main APP-003          # runs a specific application ID
python -m src.main --all            # runs all 6 test applications
```

### Run the end-to-end demo (all 6 cases)

```bash
python scratch/run_demo_cases.py            # PDF parsing mode (exercises Phase 4 pipeline)
python scratch/run_demo_cases.py --raw      # Fast mode using embedded text from JSON
```

Expected output:
```
Demo Summary: 6/6 cases verified successfully.
```

### Launch the Streamlit officer dashboard

```bash
streamlit run src/ui/officer_dashboard.py
# or:
make run-demo
```

---

## Testing

```bash
# Run the full test suite (123 tests)
make test

# Verbose with coverage report
pytest -v --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_kyc.py -v
pytest tests/test_observability.py -v
```

### Test suite breakdown

| Test file | Tests | Coverage area |
|-----------|-------|---------------|
| `test_credit.py` | 26 | Credit scoring, DTI, risk classification, schema |
| `test_kyc.py` | 6 | Document validation, fraud detection, income normalisation |
| `test_resilience.py` | 18 | Graceful fallbacks, 30s timeouts, all 4 agent types |
| `test_policy.py` | 6 | Policy retrieval, LTV/DTI checks, RAG integration |
| `test_guardrails.py` | 7 | Input validation, output confidence guards, audit logging |
| `test_graph.py` | 5 | End-to-end graph runs, fraud routing, all 6 test cases |
| `test_arbitrator.py` | 8 | Unanimous/conflict/fraud/missing-docs scenarios |
| `test_arbitrator_dummy.py` | 8 | Arbitrator edge cases |
| `test_edges.py` | 8 | Graph routing logic (KYC → credit, KYC → review, etc.) |
| `test_observability.py` | 11 | LangSmith tracing, env vars, trace_id propagation |
| `test_human_review_tool.py` | 7 | HITL tool, override validation |
| `test_state.py` | 5 | State schema serialisation, decorator validation |
| `test_integration.py` | ... | Full pipeline integration, parametrised risk levels |
| `test_document_tools.py` | 7 | PDF parsing, OCR fallback, field extraction |

---

## Observability

Every run generates a LangSmith trace capturing all agent nodes, inputs, outputs, and latencies.

### View traces

1. Ensure `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set in `.env`
2. Run any pipeline command (e.g. `python -m src.main`)
3. The trace URL is printed to stdout:
   ```
   [trace] View trace: https://smith.langchain.com/o/.../projects/p/RiskPilot/r/<run-id>
   ```
4. Open the URL in your browser to see the full node-by-node execution graph

### Local audit log

Every validation flag, recommendation, and human decision is also written to `logs/audit.jsonl`:

```bash
cat logs/audit.jsonl | python -m json.tool | head -40
```

---

## Available Make Commands

| Target | Description |
|--------|-------------|
| `make install` | Install dependencies + pre-commit hooks |
| `make test` | Run full pytest suite with coverage |
| `make run-demo` | Launch Streamlit officer dashboard |
| `make lint` | Check code style (black, isort, flake8) |
| `make format` | Auto-format code (black + isort) |
| `make clean` | Remove `__pycache__`, `.coverage`, `chroma_db` |

---

## Team & Phase Allocation

| Member | Phases | Domain |
|--------|--------|--------|
| **A** | 1, 4, 5, 10, 14, 16 | Data pipeline, KYC agent, synthetic data, input guardrails, resilience |
| **B** | 2, 6, 11, 15, 19 | State schema, credit scoring, output guards, integration testing |
| **C** | 3, 7, 17, 20 | RAG pipeline, policy agent, performance, general guidelines |
| **D** | 8, 9, 12, 13 | Arbitrator, graph orchestration, HITL UI, observability |

---

## PRD Compliance Checklist

| Requirement | Status |
|-------------|--------|
| 4-agent pipeline (KYC, Credit, Policy, Arbitrator) | ✅ |
| LangGraph orchestration with conditional routing | ✅ |
| Human-in-the-loop — mandatory officer sign-off | ✅ |
| RAG-backed policy retrieval (6 policy docs) | ✅ |
| Pydantic-typed `LoanApplicationState` | ✅ |
| Input guardrails (schema, doc count, file type/size) | ✅ |
| Output guardrails (confidence threshold, schema) | ✅ |
| Fraud detection (name + income mismatch) | ✅ |
| Graceful fallbacks on agent failure | ✅ |
| 30s timeout on parse/extraction layer | ✅ |
| Append-only audit log (`logs/audit.jsonl`) | ✅ |
| LangSmith observability + `trace_id` per run | ✅ |
| Synthetic test data (APP-001 … APP-006) | ✅ |
| All 6 demo cases pass end-to-end | ✅ |
| 123 automated tests — 100% pass rate | ✅ |
| General underwriting guidelines policy doc | ✅ |
