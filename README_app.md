# Demo de Comparação (Streamlit)

Este app permite comparar o modelo base (TinyLlama) com o modelo fine-tunado (mesclado) ou via adapter (LoRA), gerando descrições a partir de títulos no mesmo formato de prompt usado no treinamento. 

## Requisitos
- Python 3.10+
- Windows (PowerShell) ou outro OS
- GPU opcional (acelera bastante). Em CPU funciona, porém mais lento.

Instale as dependências com o arquivo `requirements.txt`. A biblioteca `unsloth` é opcional, mas acelera e simplifica o carregamento. Em Windows, se a instalação do `unsloth` for difícil, o app usa automaticamente o fallback de Transformers padrão.

## Artefatos esperados
- Adapter (LoRA): `tinyllama_amazon_finetuned/`
- Modelo mesclado (completo): `tinyllama_amazon_final/`
- Dataset de treino usado (salvo pelo notebook): `artifacts/train_dataset_used.jsonl` e `.jsonl.gz`

## Como usar
1. Ative seu ambiente virtual e instale as dependências com `requirements.txt`.
2. Execute o app Streamlit: `streamlit run streamlit_app.py`.
3. No app (sidebar):
   - Selecione o tipo de modelo:
     - Base (HF Hub): carrega `TinyLlama/TinyLlama-1.1B-Chat-v1.0` do Hub (ou outro ID informado).
     - Fine-tuned (mesclado): aponte para a pasta local `tinyllama_amazon_final`.
     - Fine-tuned (adapter): informe o caminho do adapter em `tinyllama_amazon_finetuned` e o ID do modelo base (TinyLlama do Hub).
   - Ajuste os parâmetros de geração (max_new_tokens, temperature, top_p) conforme desejar.
4. Informe um título e clique em "Gerar descrição".

Dica: Para uso em produção, prefira o modelo mesclado para evitar dependências de adapter em tempo de execução.

## Observações
- Em GPU o app tentará usar quantização 4-bit via Unsloth quando disponível; caso contrário, usa Transformers padrão.
- Se o tokenizer não tiver token de padding, o app adiciona `<|pad|>` e ajusta as embeddings.
- O prompt segue: `[SYSTEM] ... [USER] Título: ... [ASSISTANT]` (mesmo do notebook).
