import os
import re
import json
import shutil
import hashlib
import streamlit as st
import torch
from typing import Tuple, List

# Dependências de modelo (Unsloth é opcional)
try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None
from transformers import AutoTokenizer, AutoModelForCausalLM

# Opcional para carregar adapter
try:
    from peft import PeftModel
except Exception:
    PeftModel = None

st.set_page_config(page_title="Tech Challenge - FT Demo", page_icon="🤖", layout="wide")
st.title("Tech Challenge - Fine-tuning Demo (TinyLlama)")

st.markdown(
    """
    Select a base model (HF Hub), a fine-tuned merged model, or a LoRA adapter to generate product descriptions from a title.
    The prompt follows the same format used during training.
    """
)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def _looks_like_merged_model(dir_path: str) -> bool:
    # Critérios simples: tem config.json e algum peso de modelo
    if not os.path.isdir(dir_path):
        return False
    files = set(os.listdir(dir_path))
    has_config = 'config.json' in files
    # Aceitar pesos shardeados: *.safetensors ou pytorch_model*.bin
    has_weights = any(
        f.endswith('.safetensors') or (f.startswith('pytorch_model') and f.endswith('.bin'))
        for f in files
    )
    return has_config and has_weights

def _looks_like_adapter(dir_path: str) -> bool:
    if not os.path.isdir(dir_path):
        return False
    files = set(os.listdir(dir_path))
    # Padrão PEFT
    return ('adapter_config.json' in files) or ('adapter_model.safetensors' in files)

def _discover_dirs(root: str, predicate) -> List[str]:
    found = []
    # verifica o próprio root e 1 nível de subpastas
    if predicate(root):
        found.append(root)
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and predicate(path):
            found.append(path)
    # ordenar por nome para estabilidade
    return sorted(found)

def _sanitize_adapter_dir(src_dir: str) -> str:
    """
    Some adapters trained with Unsloth/PEFT include extra keys (e.g., 'corda_config')
    that older peft versions don't accept. This creates a sanitized copy with unknown
    keys removed from adapter_config.json.
    """
    cfg_path = os.path.join(src_dir, 'adapter_config.json')
    if not os.path.exists(cfg_path):
        return src_dir
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        return src_dir

    # Determine allowed keys from the installed peft's LoraConfig signature
    allowed = None
    try:
        from peft import LoraConfig
        import inspect
        sig = inspect.signature(LoraConfig.__init__)
        allowed = set([p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD])
        # Remove 'self' if present
        allowed.discard('self')
    except Exception:
        # Fallback broad allowlist across common peft versions
        allowed = {
            'r', 'target_modules', 'lora_alpha', 'lora_dropout', 'bias', 'task_type',
            'inference_mode', 'use_rslora', 'peft_type', 'rank_pattern', 'alpha_pattern',
            'modules_to_save', 'init_lora_weights', 'fan_in_fan_out', 'layers_pattern',
            'layers_to_transform', 'target_parameters', 'megatron_config', 'megatron_core',
            'layer_replication', 'base_model_name_or_path', 'auto_mapping', 'revision',
            'use_dora', 'loftq_config', 'use_qalora', 'qalora_group_size', 'exclude_modules'
        }

    # Drop unknown keys (e.g., 'corda_config', 'eva_config')
    sanitized = {k: v for k, v in cfg.items() if k in allowed}
    if sanitized == cfg:
        return src_dir  # nothing to change

    # Make a deterministic cache folder name based on src_dir path
    h = hashlib.sha256(src_dir.encode('utf-8')).hexdigest()[:16]
    dst_dir = os.path.join(PROJECT_ROOT, '.cache_sanitized_adapters', h)
    os.makedirs(dst_dir, exist_ok=True)
    # Copy all files from src except the config we'll overwrite
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if name == 'adapter_config.json':
            continue
        if os.path.isdir(src):
            # shallow copy of subfolders if any (shouldn't be necessary for PEFT)
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    # Write sanitized config
    with open(os.path.join(dst_dir, 'adapter_config.json'), 'w', encoding='utf-8') as f:
        json.dump(sanitized, f, ensure_ascii=False, indent=2)
    return dst_dir

# Sidebar configs
with st.sidebar:
    st.header("Model Settings")
    model_mode = st.selectbox(
        "Model type",
        ["Base (HF Hub)", "Fine-tuned (merged)", "Fine-tuned (adapter)"]
    )
    base_model_name = st.text_input("HF model id (base)", value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # Descobrir diretórios na raiz do projeto
    merged_candidates = _discover_dirs(PROJECT_ROOT, _looks_like_merged_model)
    adapter_candidates = _discover_dirs(PROJECT_ROOT, _looks_like_adapter)

    st.caption(f"Project root: {PROJECT_ROOT}")

    if merged_candidates:
        merged_model_path = st.selectbox("Merged model (auto-detected)", merged_candidates, index=0)
    else:
        merged_model_path = st.text_input("Merged model path", value=os.path.join(PROJECT_ROOT, "tinyllama_amazon_final"))

    if adapter_candidates:
        adapter_path = st.selectbox("LoRA adapter (auto-detected)", adapter_candidates, index=0)
    else:
        adapter_path = st.text_input("LoRA adapter path", value=os.path.join(PROJECT_ROOT, "tinyllama_amazon_finetuned"))

    max_seq_len = st.number_input("Max seq len", min_value=256, max_value=4096, value=1024, step=128)

    st.header("Generation")
    max_new_tokens = st.slider("max_new_tokens", 16, 512, 128, 8)
    do_sample = st.checkbox("do_sample", value=True)
    temperature = st.slider("temperature", 0.1, 1.5, 0.7, 0.1)
    top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.05)

SYSTEM_PROMPT_DEFAULT = "You are an assistant that writes detailed product descriptions from a given title."

@st.cache_resource(show_spinner=False)
def load_model(model_mode: str, base_model_name: str, merged_model_path: str, adapter_path: str, max_seq_len: int):
    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'
    )
    load_in_4bit = True if (device == 'cuda' and FastLanguageModel is not None) else False

    def _unsloth_load(model_id_or_path: str):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id_or_path,
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        return model, tokenizer

    def _hf_load(model_id_or_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        try:
            torch_dtype = torch.float16 if device == 'cuda' else None
            model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                torch_dtype=torch_dtype,
                device_map='auto' if device == 'cuda' else None,
            )
        except Exception:
            # Fallback CPU
            model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
            model.to(device)
        if len(getattr(model, 'get_input_embeddings')().weight) != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    # Base
    if model_mode == "Base (HF Hub)":
        if FastLanguageModel is not None:
            return _unsloth_load(base_model_name)
        else:
            st.warning("Unsloth não encontrado; usando Transformers padrão (pode ser mais lento).")
            return _hf_load(base_model_name)

    # Mesclado
    if model_mode == "Fine-tuned (merged)":
        if not os.path.isdir(merged_model_path):
            st.error(f"Diretório de modelo mesclado não encontrado: {merged_model_path}")
            st.stop()
        if not _looks_like_merged_model(merged_model_path):
            st.error(
                "O diretório selecionado não contém pesos do modelo.\n"
                "Certifique-se de que os arquivos de pesos (*.safetensors ou pytorch_model*.bin) estão DENTRO desta pasta, junto com o config.json.\n"
                "Observação: você colocou 'model.safetensors' na raiz do projeto — mova-o para a pasta do modelo mesclado (ex.: tinyllama_amazon_final).\n"
                "Alternativa: reabra o notebook e execute a etapa de MERGE & SAVE para salvar tudo na mesma pasta."
            )
            st.stop()
        if FastLanguageModel is not None:
            return _unsloth_load(merged_model_path)
        else:
            st.warning("Unsloth não encontrado; usando Transformers padrão (pode ser mais lento).")
            return _hf_load(merged_model_path)

    # Adapter
    if PeftModel is None:
        st.error("peft is not installed. Install peft to load adapters, or use the merged model.")
        st.stop()
    if not os.path.isdir(adapter_path):
        st.error(f"Diretório de adapter não encontrado: {adapter_path}")
        st.stop()
    if FastLanguageModel is not None:
        base_model, tokenizer = _unsloth_load(base_model_name)
    else:
        st.warning("Unsloth não encontrado; usando Transformers padrão (pode ser mais lento).")
        base_model, tokenizer = _hf_load(base_model_name)
    # Try raw adapter first; if it fails due to extra config keys, sanitize and retry
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        # Try sanitization when error mentions unexpected keyword arg
        if 'unexpected keyword argument' in str(e) or 'got an unexpected keyword' in str(e):
            sanitized_dir = _sanitize_adapter_dir(adapter_path)
            try:
                model = PeftModel.from_pretrained(base_model, sanitized_dir)
            except Exception as e2:
                st.error(
                    "Failed to load adapter even after sanitizing adapter_config.json.\n"
                    f"Original error: {e}\nRetry error: {e2}\n"
                    "Solução: atualize o pacote 'peft' (>=0.11) ou use o modo 'Fine-tuned (merged)'."
                )
                st.stop()
        else:
            st.error(f"Failed to load adapter: {e}")
            st.stop()
    return model, tokenizer

@torch.no_grad()
def generate(model, tokenizer, title: str, system_prompt: str, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float, max_seq_len: int) -> str:
    prompt = f"[SYSTEM]\n{system_prompt}\n[USER]\nTitle: {title}\n[ASSISTANT]\n"
    inputs = tokenizer([prompt], return_tensors='pt', padding=True, truncation=True, max_length=max_seq_len)
    # Move para o device correto
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = (inputs['input_ids'] != (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)).sum(dim=1)[0]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove o prompt
    prompt_text = tokenizer.decode(outputs[0][:input_len], skip_special_tokens=True)
    return text[len(prompt_text):].strip()

with st.form("gen_form"):
    st.subheader("Generate description from title")
    system_prompt = st.text_area("System prompt", value=SYSTEM_PROMPT_DEFAULT, height=100)
    title_text = st.text_input("Product title", value="Smartwatch with GPS and heart-rate monitor")
    submitted = st.form_submit_button("Generate")

if submitted:
    with st.spinner("Carregando modelo e gerando..."):
        try:
            model, tokenizer = load_model(model_mode, base_model_name, merged_model_path, adapter_path, max_seq_len)
            model.eval()
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
                model.resize_token_embeddings(len(tokenizer))
            output = generate(model, tokenizer, title_text, system_prompt, max_new_tokens, do_sample, temperature, top_p, max_seq_len)
            st.success("Generation completed!")
            st.write("### Generated description:")
            st.write(output)
        except Exception as e:
            st.error(f"Error during generation: {e}")

st.caption("Tip: prefer the merged model for production to avoid adapter dependencies. On GPU, 4-bit may be enabled automatically.")