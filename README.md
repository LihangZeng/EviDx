# EviDx

**Evidence-Aware Active Diagnosis with Scaffolded LLM Agents**

[Paper] | [Med-Evidence-2.6k](https://huggingface.co/datasets/danieez/Med-Evidence-2.6k) | [Apache-2.0](LICENSE)

EviDx is a research framework for active clinical diagnosis with LLM agents. It turns static clinical cases into interactive patient-specific environments, lets agents acquire patient and external evidence through tools, and regulates diagnostic termination using uncertainty and evidence coverage. The paper was accepted to *Findings of EMNLP 2026*.

## Framework

EviDx contains four main components:

- **E-Synthesis** converts raw cases into structured, queryable EHR environments.
- **Clinical Dx Scaffold** coordinates the Diagnostician, EHR Executor, Clinical Consultant, evidence tools, and evolving diagnostic states.
- **Observer-Guided Harness** monitors trajectories and controls termination using diagnostic uncertainty and evidence coverage.
- **3-Level Evaluation Pyramid** evaluates execution robustness, reasoning dynamics, and diagnostic outcomes.

```text
Clinical case -> E-Synthesis -> Diagnostician
                                  |-- EHR Executor
                                  |-- Clinical Consultant
                                  `-- Observer-Guided Harness
                               -> Diagnosis and trajectory evaluation
```

## Repository

```text
MedEvidence2k6_annotation/   Evidence annotation pipeline
rag/                         Qdrant knowledge-base builder
agent/Synth/                 Case-to-EHR synthesis
agent/Diag/                  Diagnostic reasoning agent
agent/Exam/                  Patient-record tools
agent/Consult/               External-knowledge consultation
agent/Observe/               Runtime monitoring and termination
agent/Judge/                 Diagnostic evaluation
```

## Setup

Python 3.10 or newer and Docker are recommended.

```bash
conda create -n evidx python=3.10 -y
conda activate evidx
pip install -r requirements.txt
cp .env.example .env
```

Configure `.env` with:

- an OpenAI-compatible LLM endpoint;
- an OpenAI-compatible embedding endpoint;
- a Qdrant instance;
- an optional directory of [MedRAG](https://huggingface.co/MedRAG) textbook JSONL files for BM25 retrieval.

See [.env.example](.env.example) for all variables. The StatPearls and textbook corpora are available from [MedRAG](https://huggingface.co/MedRAG); model weights are not included.

## Usage

Start a local Qdrant service:

```bash
docker pull qdrant/qdrant
docker run --name evidx-qdrant -d \
  -p 127.0.0.1:6333:6333 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

Start the OpenAI-compatible embedding service configured by `EMBED_*`, then download the [MedRAG](https://huggingface.co/MedRAG) StatPearls corpus and build the dense knowledge base:

```bash
python rag/build_knowledge_base-linux.py path/to/statpearls.jsonl
```

Run active diagnosis:

```bash
python agent/main.py \
  --input-file path/to/cases.jsonl \
  --max-cases 100 \
  --output-dir logs
```

Evaluate evidence acquisition and use:

Download [Med-Evidence-2.6k](https://huggingface.co/datasets/danieez/Med-Evidence-2.6k) and pass its JSONL file to the evaluator. The released annotations are ready to use; rerunning the annotation pipeline is not required.

```bash
python agent/evaluate_evidence_recall.py \
  --trace_file logs/inference_traces_TIMESTAMP.jsonl \
  --golden_files path/to/Med-Evidence-2.6k \
  --output_dir eval_results
```

## Data

[Med-Evidence-2.6k](https://huggingface.co/datasets/danieez/Med-Evidence-2.6k) contains 2,660 LLM-assisted reference-evidence annotations. Each annotation links diagnostic evidence to the source case and categorizes its role as inclusion, exclusion, or differentiation.

The source cases come from:

- [DiagnosisArena](https://huggingface.co/datasets/SII-SPIRAL-MED/DiagnosisArena)
- [MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA)
- [JAMA Clinical Challenge](https://huggingface.co/datasets/JesseLiu/Jama_challenge)

The StatPearls and textbook retrieval corpora come from [MedRAG](https://huggingface.co/MedRAG). 

## License

The source code and documentation are released under the [Apache License 2.0](LICENSE). This license does not cover third-party datasets, corpora, or model weights.

## Disclaimer

EviDx is for research only. It is not a medical device and must not be used for clinical diagnosis, treatment, or patient-care decisions.
