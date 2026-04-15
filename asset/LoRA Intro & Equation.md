<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# **LoRA Intro \& Equation**[11]

## **1. Introduction (50 words)**

**LoRA** (**Low-Rank Adaptation**) fine-tunes **large pre-trained models** by **freezing original weights (W₀)** and injecting **tiny low-rank matrices (A×B)** into **attention layers** (Q, K, V, O). **~0.1-1% trainable params**, **10-100x memory savings**, **matches full fine-tune performance**. Perfect for **edge deployment** of **FLAN-T5** summarization.[^1][^3]

## **2. Core Equation**

```
**Original**: h = W₀ × x

**LoRA Adapted**: h = W₀ × x + (α/r) × B × A × x

Where:
- W₀ ∈ ℝ^(d×k)  = Frozen pretrained weights
- B ∈ ℝ^(d×r)    = Train **down** projection (initialized 0)
- A ∈ ℝ^(r×k)    = Train **up** projection (random normal)
- r ≪ min(d,k)   = **Low rank** (e.g., r=128 vs d=k=4096)
- α/r            = **Scaling** factor (stabilizes magnitude)
```

**Forward Pass**: During inference, **merge** B×A into W₀ → **single standard weight matrix**.

**Your notebook**: **r=128**, **α=64** → **~28MB adapter** vs **500MB full model**.[^2][^11]

## **Key Takeaways**

**LoRA = low-rank hack**: Train **r×(d+k) params** (1.2M) instead of **d×k** (16M+). **Deploy as normal HF model**. Your **41.8% ROUGE-1** proves **<1% params = 76% full FT**.[^11]

Hope this helps! Let me know if you have any other questions!
<span style="display:none">[^10][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://www.ibm.com/think/topics/lora

[^2]: https://apxml.com/courses/lora-peft-efficient-llm-training/chapter-2-lora-in-depth/lora-mathematical-formulation

[^3]: https://en.wikipedia.org/wiki/LoRA_(machine_learning)

[^4]: https://minimatech.org/deep-dive-into-lora/

[^5]: https://www.cloudflare.com/learning/ai/what-is-lora/

[^6]: https://towardsdatascience.com/understanding-lora-low-rank-adaptation-for-finetuning-large-models-936bce1a07c6/

[^7]: https://www.docker.com/blog/lora-explained/

[^8]: https://www.youtube.com/watch?v=Bq9zqTJDsjg

[^9]: https://www.geeksforgeeks.org/deep-learning/what-is-low-rank-adaptation-lora/

[^10]: https://www.entrypointai.com/blog/lora-fine-tuning/

[^11]: LoRa_fine_tune.ipynb



---

📊 ROUGE Metrics (Evaluation for Summarization)

ROUGE = Recall‑Oriented Understudy for Gisting Evaluation
It measures how much overlap exists between a generated summary and a reference (ground truth) summary.
🔹 ROUGE‑1

Definition:
Measures overlap of unigrams (single words) between generated and reference summaries.

Formula (Recall version):
ROUGE-1=Number of overlapping unigramsTotal unigrams in reference
ROUGE-1=Total unigrams in referenceNumber of overlapping unigrams​



ROUGE‑2

Definition:
Measures overlap of bigrams (two consecutive words).

Formula (Recall version):
ROUGE-2=Number of overlapping bigramsTotal bigrams in reference
ROUGE-2=Total bigrams in referenceNumber of overlapping bigrams​


ROUGE‑L

Definition:
Based on Longest Common Subsequence (LCS) between generated and reference text.

LCS = longest sequence of words appearing in both texts in the same order (not necessarily consecutive).

Formula (Recall version):
ROUGE-L=Length of LCSLength of reference
ROUGE-L=Length of referenceLength of LCS​


Intuition

    BLEU asks:(Precision-Oriented)
    “Did you generate exactly what I expected?”

    ROUGE asks: (Recall-Oriented)
    “Did you cover the important parts?”



![Zero Shot](asset/1_zero-shot2.png)

![Full Fine Tune](asset/2_full-fine-tune.png)

![LoRA Optimization](asset/3_LoRa-Optimization.png)

![Quantitative Evaluations](asset/4_QEvals.png)

![LoRA Demo](asset/5_LoRa_Demo.png)

![Encoder Decoder](asset/Encoder_decoder.png)

![LoRA Internal](asset/Lora.jpg)