<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Advanced FLAN‑T5 + LoRA Fine‑Tuning Project Roadmap for Finance / Fintech

This document outlines a **step‑by‑step, phase‑based plan** to upgrade your existing FLAN‑T5 247M + LoRA project into an **advanced‑level, finance‑oriented** summarization system suitable for fintech or financial‑services applications.

***

## Phase 1 – Problem Reframing \& Finance‑Domain Setup

### Objective

Reframe dialogue‑summarization into a **finance‑style, risk‑aware** task and prepare domain‑specific data.

### Deliverables

- Finance‑themed dataset schema.
- Clear problem statement and target user (e.g., traders, IR, support).


### Tasks

1. **Define the fintech use case**
    - Example use cases:
        - **Earnings‑call / trader–client dialogue summarization**
        - **Support‑ticket or risk‑note summarization for operations**
    - Choose one primary use case to start.
2. **Create finance‑style sample texts**
    - Convert your current `dialuge_summarize` pairs into:

```text
Context: Trader–client call about AAPL earnings...
Dialogues (turns between trader and client)
Summary (short, risk‑aware, 1–2 sentences)
```

    - Add metadata:
        - `domain: "earnings_call"`, `"trading"`, `"client_support"`
        - `tags: ["risk", "compliance"]` (optional)
3. **Dataset versioning**
    - Keep versions:
        - `v0` – generic dialogues (current)
        - `v1` – finance‑style dialogues, aligned with above schema
    - Store in a structured folder:

```bash
data/
  v0_dialogue_summarize/
  v1_finance_summarize/
```

4. **Prompt engineering**
    - Design a **finance‑style instruction**:

```text
Summarize the following financial dialogue into a risk‑aware, client‑ready note.
...

Summary:
```

    - Use this same prompt consistently across zero‑shot, full‑fine‑tune, and LoRA experiments.

***

## Phase 2 – Model Training \& Optimization (Beyond ROUGE)

### Objective

Upgrade your FLAN‑T5 247M baselines (zero‑shot, full‑fine‑tune, LoRA) with **fin‑tuned LoRA + multi‑objective learning** and better hyperparameter tuning.

### Deliverables

- Multiple LoRA‑tuned checkpoints.
- Hyperparameter sweep logs (Optuna).
- Domain‑aware, multi‑objective loss.


### Tasks

1. **Establish baselines**
    - Zero‑shot FLAN‑T5 (with your finance‑prompt).
    - Full fine‑tune on finance‑style data.
    - LoRA‑baseline (no Optuna, default `r`, `alpha`, `dropout`).
2. **Enhance LoRA‑Optuna experiments**
    - Search space:
        - `r` ∈ `[8, 16, 32, 64, 128]`
        - `alpha` ∈ `[8, 16, 32, 64]`
        - `dropout` ∈ `[0.01, 0.1, 0.2]`
        - `lr` ∈ `[1e‑5, 5e‑4]`
        - `batch_size` ∈ `[8, 16, 32]`
    - Metrics:
        - `primary: rougeL_fmeasure`
        - `secondary: rouge1_fmeasure, fact_consistency_score`
    - Use Optuna to maximize:

$$
\text{objective} = \alpha \cdot \text{ROUGE-L} + (1 - \alpha) \cdot \text{Fact‑Consistency}
$$
3. **Add multi‑objective training**
    - Add a small **risk‑level classifier head** on top of the encoder:
        - Labels: `low`, `medium`, `high` risk (based on reference summaries).
    - Train LoRA with combined loss:

$$
\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{seq2seq}} + (1 - \lambda) \cdot \mathcal{L}_{\text{risk‑level}}
$$
    - Validate that the model both:
        - Produces high‑quality summaries (ROUGE‑L).
        - Predicts consistent risk levels.
4. **Checkpoint management**
    - Save checkpoints with:
        - `version` (e.g., `lora_v1`, `lora_v2`, `lora_optuna_best`)
        - `config` (JSON: `r`, `alpha`, `dropout`, `lr`, `batch_size`)
    - Register the best‑performing model as the **production‑ready checkpoint**.

***

## Phase 3 – Advanced Evaluation \& Metrics

### Objective

Go beyond ROUGE and add **domain‑aware, fintech‑style evaluation** (quantitative + qualitative).

### Deliverables

- Quantitative metrics table (per model).
- Qualitative evaluation sheet and Streamlit‑based UI for human‑ranking.
- Batch‑processing evaluation loop.


### Tasks

1. **Domain‑specific quantitative metrics**
    - Implement:
        - **Entity‑preservation**:
            - Extract entities: tickers, currency amounts, percentages, dates.
            - Compute % of entities preserved in summary vs reference.
        - **Fact‑consistency**:
            - If reference mentions “-2.3%”, summary must not say “+2.3%”.
            - Compute % of key numbers correctly preserved.
    - Measure for:
        - Zero‑shot
        - Full fine‑tune
        - LoRA‑various, LoRA‑optuna‑best
2. **Table of results**
    - Show per‑model:


| Model | ROUGE‑1 | ROUGE‑L | Entity‑P | Fact‑Cons |
| :-- | :-- | :-- | :-- | :-- |
| Zero‑shot FLAN‑T5 | 0.32 | 0.28 | 0.60 | 0.72 |
| Full‑fine‑tune | 0.41 | 0.36 | 0.75 | 0.81 |
| LoRA‑optuna‑best | 0.40 | 0.37 | 0.78 | 0.84 |

3. **Structured human evaluation**
    - Create a **qualitative‑evaluation sheet** (Google Form / internal sheet) with questions like:
        - “Does the summary preserve the risk direction?”
        - “Are numbers and currencies correct?”
        - “Is the tone appropriate for client‑facing use?”
        - Score 1–5 on each.
    - Run side‑by‑side evaluation for:
        - Zero‑shot vs full‑fine‑tune vs LoRA‑optuna‑best
4. **Integrate qualitative UI into Streamlit**
    - Add a **Comparison View** in `streamlit_app.py`:
        - Single‑input dialogue, 3–4 model outputs in columns.
        - Side‑by‑side summaries, with:
            - ROUGE scores
            - Risk‑level badge
            - Fact‑consistency indicator
    - Optionally:
        - Add a **“Rank” button** that sends feedback to an external sheet or DB.

***

## Phase 4 – Fintech‑Style Streamlit UI \& Serving

### Objective

Turn your Streamlit app into a **fintech‑style analytics dashboard** with risk‑tagging, batch‑mode, and model‑version control.

### Deliverables

- `streamlit_app.py` re‑structured as:
    - Single‑query view
    - Batch‑mode ingest
- Model‑version picker and risk‑level outputs.


### Tasks

1. **Model‑version selector**
    - In Sidebar, add:

```python
model_version = st.sidebar.selectbox(
    "Model",
    ["zero_shot", "full_finetune", "lora_optuna_best"]
)
```

    - Load the appropriate checkpoint when `model_version` changes.
2. **Risk‑level and fact‑consistency tags**
    - For each generated summary, show:
        - `Risk‑level` badge (🟢 Low, 🟡 Medium, 🔴 High).
        - `Fact‑Consistency: 92%` if available.
    - Color‑code the summary box accordingly.
3. **Multi‑model comparison view**
    - For a single input, show:
        - 3–4 columns, one per model version.
        - Each column shows:
            - Model name
            - Summary text
            - ROUGE‑1, ROUGE‑L, Fact‑Cons
    - This converts your UI into a **fintech‑style A/B testing dashboard**.
4. **Batch‑mode processing**
    - Add a **Batch Tab**:
        - Text area to paste many dialogues (one per line).
        - Button: **“Run in Batch”**.
    - Output:
        - A table with:
            - Dialogue snippet
            - Selected model summary
            - ROUGE‑1, ROUGE‑L, entity‑preservation, fact‑consistency
        - Download as CSV:

```python
st.download_button("Download CSV", csv_buffer, "batch_results.csv")
```

5. **Monitoring view (optional advanced)**
    - Add a **Monitoring Dashboard** tab:
        - Upload a small “monitoring” set (e.g., weekly earnings‑call dialogues).
        - Run best model on them.
        - Show:
            - Average ROUGE decay over time.
            - Average fact‑consistency trend.
        - This looks like a **production‑style monitoring dashboard**.

***

## Phase 5 – MLOps / Pipeline / Fintech‑Ready Design

### Objective

Make the project look **production‑ready** and suitable for fintech infrastructure (logging, monitoring, versioning).

### Deliverables

- Experiment logging setup (e.g., MLflow / W\&B).
- Lightweight API wrapper or `ServeStreamlit` config.
- Simple data‑drift / monitoring dashboard.


### Tasks

1. **Experiment logging**
    - Use `MLflow` or `Weights & Biases`:
        - Log each Optuna trial:
            - `r`, `alpha`, `dropout`, `lr`, `batch_size`
            - `rouge1`, `rougeL`, `fact_consistency`
        - Register the best‑performing LoRA config as `flan-t5-lora-finance-247M-v1`.
2. **Model serving**
    - Option A: Lightning `ServeStreamlit`
        - Wrap your LoRA‑fine‑tuned model so it can be deployed as a **web‑accessible endpoint**.
    - Option B: Small FastAPI / Flask wrapper:
        - Input:

```json
{"dialogue": "...", "model_version": "flan-t5-lora-finance-v1"}
```

        - Output:

```json
{
  "summary": "...",
  "risk_level": "medium",
  "fact_consistency": 0.92
}
```

3. **Data‑drift / model‑monitoring concept**
    - Once you have a “production‑ready” model, periodically:
        - Ingest new dialogues (e.g., from earnings‑call feeds).
        - Run the model and store:
            - Predicted summary
            - Risk‑level
            - ROUGE vs reference (if available)
        - Compute trends:
            - Is ROUGE‑L slowly decreasing over time?
            - Is fact‑consistency decaying?
    - Visualize this in Streamlit or a separate dashboard.
4. **Documentation \& portfolio framing**
    - Write a short report:
        - Problem: “Risk‑aware summarization of finance‑style dialogues for traders, IR, and support teams.”
        - Method:
            - FLAN‑T5 247M with LoRA + Optuna hyperparameter tuning.
            - Multi‑objective training (summary quality + risk‑level).
            - Fact‑consistency‑aware metrics.
        - UI:
            - Single‑query + batch‑mode summarization.
            - Multi‑model comparison with risk‑tagging.
    - Highlight:
        - How this reduces manual note‑writing.
        - How it flags risky or inconsistent summaries for human review.

***

This roadmap gives you a **clear, phase‑by‑phase** guide to advance your project from a research‑grade demo into a **fintech‑ready, production‑oriented system** that you can showcase in your portfolio or research outputs.

