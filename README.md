# **LoRA Fine-Tune Optimization: Dialogue Summarization with FLAN-T5-Base **

**Profile Description**: Production-grade **LoRA fine-tuning pipeline** for **FLAN-T5-Base (247M params)** on **DialogSum dataset**, achieving **ROUGE-1 ~41.8%** via **Optuna HPO**. Compares **zero-shot**, **full fine-tune (42% ROUGE-1)**, and **LoRA (30-41% ROUGE-1, 94% param efficiency)**. Runs on **Lightning.AI L40S/A40** with **PEFT**, **TRL Trainer**, **bf16**. Saves **merged model** (~500MB) + **LoRA adapter** (28MB) for deployment.[^1]

## **1. Model Details**

- **Base Model**: **google/flan-t5-base/google** (247M params, encoder-decoder seq2seq)
- **Architecture**: **FLAN-T5** (text-to-text framework, instruction-tuned for summarization/QA/classification)
- **LoRA Config** (best trial): **r=128**, **alpha=64**, **dropout=0.034**, **target_modules=["q", "v"]** (attention projections)
- **Param Efficiency**: **~0.5% trainable** (1.2M vs 247M full); merged model deployable as standard HF
- **Hardware**: **Lightning.AI L40S (46GB)**, **CUDA 12.1**, **bf16** precision[^1]


--- 
![Encoder Decoder](asset/Encoder_decoder.png)
--- 

## **2. Dataset Details**

- **Dataset**: **knkarthick/dialogsum** (Hugging Face)
    - **Train**: 12,460 dialogues + human summaries
    - **Validation**: 500 examples
    - **Test**: 1,500 examples
- **Task**: **Abstractive summarization** ("Summarize the following conversation: [dialogue]")
- **Preprocessing**:


| Step | Details |
| :-- | :-- |
| Tokenization | **FLAN-T5 tokenizer**, max_input=512, max_output=128 |
| Padding/Truncation | Batched, PT tensors |
| Prompt | "Summarize the following conversation." + dialogue [^1] |


## **3. Training Methods**

- **Baselines**:
    - **Zero-shot**: Original FLAN-T5 (**ROUGE-1: 24.6%**)
    - **Full Fine-Tune**: All 247M params, 20 epochs, **lr=5e-4**, **bs=4** (GA=4), cosine LR (**ROUGE-1: 42.1%** upper bound)

---

![Zero Shot](asset/1_zero-shot2.png)
---

![Full Fine Tune](asset/2_full-fine-tune.png)
---

- **LoRA Optimization**:
    - **Optuna (10 trials)**: TPE sampler, maximize **ROUGE-1**
    - **Hyperparams**: r∈, alpha∈, dropout∈[0.03-0.05], epochs∈[3-5], lr∈[5-7e-4], bs∈[^2][^3][^4]
    - **Best Config**: **r=128/alpha=64/dropout=0.034/lr=5.58e-4/epochs=3/bs=32/warmup=13.6%**
    - **Trainer**: **HF Trainer** + **EarlyStopping(patience=3)**, **eval_steps=10-100**, **max_steps=80-100/trial**
- **Efficiency**: **~94% fewer params**, **2-3x faster** than full FT[^1]

---
![LoRA Internal](asset/Lora.jpg)
---

## **4. Evaluations**

- **Metric**: **ROUGE** (1/2/L, stemmer=True, 50-100 test samples)
- **Quantitative Results** (Test Set):


| Model | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** |
| :-- | :-- | :-- | :-- |
| **Zero-Shot** | **24.6%** | **6.7%** | **21.1%** |
| **Full FT** | **42.1%** | **15.2%** | **33.2%** |
| **LoRA Best** | **30.4%** | **10.0%** | **26.7%** |

- **Qualitative**: LoRA captures key entities/actions (e.g., "Person1 suggests public transport") vs human[^1]
- **Artifacts**: **optuna_trials.csv**, **rougecomparison.csv**, **qualitativecomparison.csv**


## **5. Final Achievements**

- **ROUGE-1 Peak**: **41.8%** (Trial 7), **76% of full FT** with **<1% params**
- **Speedup**: **10 trials in <2hrs** on A40/L40S
- **Artifacts Generated**:
    - **LoRA Adapter**: `.output/lorabestmodel/` (28MB adapter.safetensors)
    - **Merged Model**: `.output/outputmerged/` (~500MB, ZIP ready)
    - **Optuna Plots**: HTML visualizations (optimization_history.html)
- **Deployment Ready**: Load via `PeftModel.from_pretrained()` or merged HF pipeline[^1]

---
![LoRA Optimization](asset/3_LoRa-Optimization.png)

---

![Quantitative Evaluations](asset/4_QEvals.png)
---

## **6. Conclusions**

**LoRA achieves 76% of full fine-tune ROUGE** with **99% param reduction**, ideal for **edge/production** on resource-constrained setups. **Optuna HPO** critical for SOTA ranks/lr. Next: **DPO/RLHF**, **longer epochs (500 steps)**, **multi-task** (QA + summary). **Reproducible** on **Lightning.AI**/**Colab Pro**.[^1]

**Run Time**: ~4hrs end-to-end. **License**: Apache 2.0. **Repo**: GitHub-ready with notebook + outputs.

<div align="center"></div> 


---

![LoRA Demo](asset/5_LoRa_Demo.png) 
---   




