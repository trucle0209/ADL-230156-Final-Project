# MATH CHILL ADVISOR

**Advanced Deep Learning — Fall 2026 — Final Project**

**Student:** Lê Nguyễn Thanh Trúc  
**Student ID:** 230156  
**Course:** CS311 — Advanced Deep Learning  
**University:** Fulbright University Vietnam

## 1. Project Overview

MATH CHILL ADVISOR is a local Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about the Applied Mathematics Major Handbook at Fulbright University Vietnam.

The project extends a text-based RAG application to support PDF documents, implements an improved hybrid retrieval method, evaluates question-answering performance, and provides both Docker deployment and a Streamlit web interface.

The chatbot uses a local language model through Ollama, without requiring a paid external LLM API.

## 2. Technologies

| Component | Technology |
|---|---|
| Programming language | Python 3.11 |
| Language model | Qwen2.5 1.5B |
| Embedding model | nomic-embed-text |
| LLM runtime | Ollama |
| RAG framework | LangChain |
| Vector database | ChromaDB |
| Retrieval methods | Semantic Search and Hybrid Retrieval (Semantic + BM25) |
| Web interface | Streamlit |
| Containerization | Docker and Docker Compose |

## 3. Main Features

- PDF document loading and text chunking.
- Semantic search and BM25-based hybrid retrieval.
- Question answering based on the Applied Mathematics Major Handbook.
- Source-page references and retrieved evidence display.
- Multi-turn conversation history.
- Translation of previous chatbot responses.
- Basic out-of-scope question handling.
- Configurable LLM model, retrieval Top-K, and timeout settings.
- Light and dark themes with customizable fonts.
- Conversation export in Markdown format.
- Docker deployment for the command-line chatbot.

## 4. Project Structure

```text
sample_code/
├── evaluation_results/
│   ├── manual_review.csv
│   ├── results.csv
│   └── results_v2.csv
│
├── major_handbook_pdfs/
│   └── MATH_MajorHandbook.pdf
│
├── report/
│   └── ADL_230156_Final_Project_Report.pdf
│
├── .dockerignore
├── .gitignore
├── chat_app.py
├── compose.yaml
├── dashboard_app.py
├── Dockerfile
├── evaluate_v2.py
├── requirements-docker.txt
├── run_pdf.py
├── run_pdf_baseline.py
├── run_txt.py
└── README.md
```

### Main Files

| File | Description |
|---|---|
| `run_txt.py` | Original text-based RAG application. |
| `run_pdf_baseline.py` | Baseline PDF RAG implementation using semantic retrieval. |
| `run_pdf.py` | Improved PDF RAG implementation using hybrid retrieval. |
| `evaluate_v2.py` | Evaluation script for the improved RAG system. |
| `dashboard_app.py` | Streamlit dashboard for reviewing evaluation results. |
| `chat_app.py` | MATH CHILL ADVISOR web chatbot. |
| `Dockerfile` | Instructions for building the Docker image. |
| `compose.yaml` | Docker Compose configuration. |
| `requirements-docker.txt` | Python dependencies for the containerized application. |

## 5. Installation

### Prerequisites

Install the following software:

- Python 3.11
- Ollama
- Docker Desktop (for Docker deployment)
- Git (for version control)

### Step 1: Download Ollama Models

Ensure Ollama is installed and running.

```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
```

The default language model is `qwen2.5:1.5b`. The smaller model is available as an alternative in the web interface.

### Step 2: Set Up the Python Environment

A separate Conda environment is recommended.

```bash
conda create -n adl-rag python=3.11 -y
conda activate adl-rag
```

### Step 3: Install Dependencies

From the project directory, run:

```bash
pip install -r requirements-docker.txt
```

For the Streamlit applications, ensure Streamlit is also installed:

```bash
pip install streamlit pandas
```

### Step 4: Prepare the PDF

The source handbook should be located at:

```text
major_handbook_pdfs/MATH_MajorHandbook.pdf
```

The application processes this document and creates a local vector database during indexing. The generated vector database is excluded from Git version control.

## 6. Running the PDF RAG Chatbot

To run the improved RAG system from the terminal:

```bash
python run_pdf.py --model qwen2.5:1.5b --top-k 5 --question "How many credits are required for the Applied Mathematics major?"
```

Expected answer:

```text
The Applied Mathematics major requires 56 credits.
```

To inspect the retrieved PDF evidence:

```bash
python run_pdf.py --model qwen2.5:1.5b --top-k 5 --show-context --question "How many credits are required for the Applied Mathematics major?"
```

The program displays the retrieved document passages and their associated PDF page numbers.

## 7. Running the Evaluation Dashboard

Launch the evaluation dashboard:

```bash
streamlit run dashboard_app.py
```

Open the local address displayed in the terminal, typically:

```text
http://localhost:8501
```

The dashboard provides evaluation metrics, baseline comparisons, individual question results, response-time statistics, and error analysis.

The evaluation results are stored in:

```text
evaluation_results/
```

## 8. Running the Web Chatbot

Launch MATH CHILL ADVISOR:

```bash
streamlit run chat_app.py
```

Open the local Streamlit URL displayed in the terminal.

The web interface supports:

- Question answering about the Applied Mathematics handbook.
- Conversation history and follow-up questions.
- Retrieved source inspection.
- Model and retrieval configuration.
- Timeout adjustment.
- Theme and font customization.
- Conversation export to Markdown.

### Example Questions

**Question 1**

```text
How many credits are required for the Applied Mathematics major?
```

Expected answer: 56 credits.

**Question 2**

```text
How about minor?
```

Expected answer: 24 credits.

**Question 3**

```text
If a student does not take Capstone I and II, what must they complete instead?
```

Expected answer: Two advanced courses at the 300-level.

**Question 4**

```text
What are the three specialized tracks in Applied Mathematics?
```

Expected answer: Data Science, Engineering Mathematics, and Quantitative Finance.

## 9. Docker Deployment

Docker is used to package and execute the command-line RAG application.

Before running Docker commands, ensure Docker Desktop and Ollama are running.

### Build the Docker Image

```bash
docker compose build
```

### Run the Containerized Chatbot

```bash
docker compose run --rm rag
```

### Verify the Docker Image

```bash
docker images
```

The Docker image is named:

```text
math-handbook-rag:local
```

The container requires access to Ollama running on the host machine. Ollama models are not packaged inside the Docker image.

The Docker demonstration uses the command-line chatbot. The Streamlit web interface is run separately.

## 10. Evaluation Results

The RAG system was evaluated on 20 questions based on the Applied Mathematics Major Handbook.

The evaluation directory contains three CSV files:

| File | Description |
|---|---|
| `results.csv` | Results from the original baseline configuration. |
| `results_v2.csv` | Results from the improved hybrid RAG configuration. |
| `manual_review.csv` | Manual answer-review records used to verify evaluation outcomes. |

### Improved RAG Performance

| Metric | Result |
|---|---|
| Total questions | 20 |
| Correct answers | 18 |
| Incorrect answers | 2 |
| Accuracy | 90% |
| Average response time | 2.606 seconds |

The improved RAG configuration achieved 90% accuracy on the 20-question evaluation set.

Two questions, Q06 and Q15, were examined further because of answer interpretation issues and differences between relevant handbook passages.

These evaluation metrics apply to the tested RAG V2 configuration. They are not a separate accuracy measurement of the final Streamlit chatbot.

## 11. Limitations

The system has several limitations:

1. The compact local language model may occasionally produce unsupported or inaccurate answers.
2. Ambiguous follow-up questions can be influenced by previous conversation topics.
3. Relevant evidence may be retrieved correctly but interpreted incorrectly during answer generation.
4. Inference performance depends on the available local hardware.
5. Questions outside the Applied Mathematics handbook may not have sufficient supporting evidence.

The web application includes basic out-of-scope handling and configurable timeout settings, but these mechanisms do not eliminate every possible error.

Users should verify important academic information against the official handbook.

## 12. Future Improvements

Potential improvements include:

- Better conversation-aware query rewriting.
- Automated verification of answers against retrieved evidence.
- Retrieval evaluation using metrics such as Recall@K.
- More comprehensive testing across different question types.
- Improved handling of ambiguous or contradictory handbook requirements.
- Support for additional academic handbooks.

## 13. Project Deliverables

The final project includes:

1. Source code for PDF-based RAG.
2. Evaluation scripts, CSV results, and dashboard.
3. Docker configuration.
4. Streamlit chatbot with conversation history.
5. Final project report in PDF format.
6. Four demonstration videos covering Tasks 1–4.

### Demonstration Videos

The following videos are prepared separately for submission:

- `task1_pdf_loading.mp4`
- `task2_rag_evaluation.mp4`
- `task3_docker_deployment.mp4`
- `task4_web_ui.mp4`

## 14. Author

**Lê Nguyễn Thanh Trúc**  
Student ID: 230156  
Advanced Deep Learning — Fall 2026  
Fulbright University Vietnam