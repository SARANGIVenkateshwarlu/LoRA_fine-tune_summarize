"""
LoRA Fine-Tune Demo - FLAN-T5 + Trial 21 Weights
Classic Streamlit App | output_merged/ folder
Venkat Sarangi | Lightning.AI Project
"""

import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# Page config - Classic look
st.set_page_config(
    page_title="LoRA Fine-Tune Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for classic look
st.markdown("""
<style>
.main {margin-top: -20px;}
h1 {color: #1f77b4; font-family: 'Arial Black';}
h2 {color: #ff7f0e;}
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
              padding: 1rem; border-radius: 10px; color: white;}
</style>
""", unsafe_allow_html=True)

# Load model from /teamspace/uploads (merged weights)
@st.cache_resource
def load_model():
    try:
        model_path = "/teamspace/uploads"  # Your merged weights folder model_path = "/teamspace/uploads" 
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
            
        )
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        st.info("Make sure `./teamspace/uploads` contains model files")
        return None, None

# ROUGE computation
@st.cache_data
def compute_rouge(pred, ref):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {k: f"{v.fmeasure:.2%}" for k, v in scores.items()}

# Title
st.markdown("""
# 🤖 **LoRA Fine-Tune Demo** 
**FLAN-T5 + Trial 21 Weights** (41.84% ROUGE-1)

*Optimized on Lightning.AI L40S 46GB*
""")

# Load model
model, tokenizer = load_model()
if model is None:
    st.stop()

# Sidebar - Model info
with st.sidebar:
    st.markdown("### 📊 **Model Stats**")
    st.info("**Architecture:** FLAN-T5-base (247M params)")
    st.success("**ROUGE Scores:**")
    st.metric("ROUGE-1", "41.84%")
    st.metric("ROUGE-2", "15.16%") 
    st.metric("ROUGE-L", "33.22%")
    st.markdown("---")
    st.markdown("**Best LoRA Params:**")
    st.code("""
r=128, alpha=64
dropout=0.034, lr=5.58e-4
epochs=3, batch_size=32
    """)

# Main app
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 **Try the Model**")
    
    # Input textarea
    prompt = st.text_area(
        "Enter conversation to summarize:",
        height=150,
        placeholder="Person1: You're finally here! What took so long?\nPerson2: I got stuck in traffic again...",
        help="Paste dialogue (up to 512 tokens)"
    )
    
    max_length = st.slider("Max output length", 50, 200, 128)
    

    if st.button("**Generate Summary** 🚀", type="primary"):
        if prompt:
            # Generate
            inputs = tokenizer(
                f"Summarize the following conversation:\n{prompt}",
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to("cpu")
                        
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Display
            st.markdown("### 📝 **Generated Summary**")
            st.success(summary)
            
            # ROUGE comparison (sample reference)
            ref_samples = [
                "Person2 arrives late due to traffic. Person1 suggests public transport.",
                "Traffic jam delays Person2. Person1 recommends alternative routes."
            ]
            rouge_scores = {ref: compute_rouge(summary, ref) for ref in ref_samples}
            
            st.markdown("### 📈 **ROUGE Scores** (vs sample refs)")
            for ref, scores in rouge_scores.items():
                col_r1, col_rl = st.columns(2)
                with col_r1:
                    st.metric("ROUGE-1", scores["rouge1"])
                with col_rl:
                    st.metric("ROUGE-L", scores["rougeL"])

with col2:
    st.markdown("### 🎯 **Demo Examples**")
    
    examples = {
        "Traffic Jam": """Person1: You're finally here! What took so long?
Person2: I got stuck in traffic again. There was a terrible traffic jam near the Carrefour intersection.
Person1: It's always rather congested during rush hour.""",
        
        "Office Memo": """Person1: Ms. Dawson, could you please take a dictation for me?
Person2: Yes, Mr. Smith.
Person1: The memo should inform all employees that instant messaging is no longer permitted.""",
        
        "Software Upgrade": """Person1: Have you considered upgrading your system?
Person2: Yes, but I'm not sure where to start.
Person1: First, update your OS, then install more RAM."""
    }
    
    selected = st.selectbox("Quick test:", list(examples.keys()))
    if selected:
        st.text_area("Auto-filled:", examples[selected], key="auto")

# Footer
st.markdown("---")
st.markdown("""
**Built with ❤️ for Venkat Sarangi**  
*LoRA Fine-Tune | Lightning.AI L40S 46GB | Trial 21 (41.84% ROUGE-1)*
""")

# Run instructions
if st.sidebar.button("ℹ️ How to Run"):
    st.sidebar.markdown("""
    ### 🚀 **Deploy Instructions**
    
    1. Save as `streamlit_app.py`
    2. Place `output_merged/` folder in same directory
    3. Terminal: `streamlit run streamlit_app.py`
    4. Open browser → **App ready!**
    
    **Files needed:**
    ```
    streamlit_app.py
    └── output_merged/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ...
    ```
    """)