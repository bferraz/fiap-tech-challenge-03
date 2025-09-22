# ⚡ OTIMIZAÇÕES APLICADAS PARA TREINO RÁPIDO

## 🎯 Principais mudanças implementadas:

### 1. **Estratégia de dados inteligente**
```python
MAX_RECORDS = 1_000_000    # Lê 1M (MÁXIMA diversidade)
ACTUAL_TRAIN_SIZE = 8_000  # Treina 8k (velocidade)
```
- **Antes**: tentando treinar em 150k+ exemplos = 8+ horas
- **Agora**: treina em 8k exemplos selecionados de 1M = 30-60min
- **Benefício**: MÁXIMA diversidade dos dados + treino ultra-rápido

### 2. **Sequências menores**
```python
OSS_MAX_SEQ_LEN = 512  # vs 1024 padrão
```
- **Economia**: ~50% menos memória e 2x mais rápido
- **Trade-off**: descrições mais concisas (ainda adequado para o domínio)

### 3. **Batch efetivo maior**
```python
OSS_BATCH_SIZE = 8      # vs 4 padrão  
OSS_GRAD_ACCUM = 4      # vs 2 padrão
# Batch efetivo = 8 × 4 = 32 (vs 8 anterior)
```
- **Economia**: 4x menos steps de treinamento
- **Benefício**: convergência mais estável com menos atualizações

### 4. **Menos épocas**
```python
OSS_EPOCHS = 1  # vs 2 padrão
```
- **Economia**: 50% menos tempo
- **Justificativa**: com 8k exemplos de qualidade, 1 época é suficiente

### 5. **Avaliação reduzida**
```python
OSS_EVAL_N = 100  # vs 200 padrão
OSS_TEST_N = 100  # vs 200 padrão
```
- **Economia**: ~50% menos tempo em avaliação
- **Baseline opcional**: pode ser pulado para ir direto ao treino

### 6. **Warmup proporcional**
```python
OSS_WARMUP_STEPS = 25  # vs 50 padrão
```
- **Ajuste**: proporcional ao número de steps reduzido

## 📊 Comparação de tempo estimado:

| Configuração | Dados lidos | Dados treino | Seq len | Batch efetivo | Épocas | Tempo estimado |
|--------------|-------------|--------------|---------|---------------|---------|----------------|
| **Original** | 150k | 150k | 1024 | 8 | 2 | **8+ horas** ⛔ |
| **Otimizada** | **1M** | 8k | 512 | 32 | 1 | **30-60min** ✅ |

## 🚀 Como usar:

1. **Execute as células em ordem** (a célula 1 já tem as configurações)
2. **Monitor o progresso**: deve mostrar steps muito menores
3. **Convergência esperada**: loss deve diminuir rapidamente
4. **Teste no Streamlit**: qualidade deve ser similar ao treino de 2k

## 💡 Se ainda estiver lento:

### Opção A: Reduzir ainda mais
```python
ACTUAL_TRAIN_SIZE = 4_000  # vs 8k
OSS_MAX_SEQ_LEN = 256      # vs 512
```

### Opção B: Pular avaliações
- Descomente apenas o bloco de treino (célula 5)
- Pule baseline, evaluation extra, etc.
- Vá direto para salvar adapter

### Opção C: Use modo T4 grátis otimizado
- No Colab: Runtime > Change runtime > T4 GPU
- Mantenha load_in_4bit=True para máxima eficiência

## ✅ Expectativa de qualidade:

Com 8k exemplos bem distribuídos:
- **Deve funcionar bem** para o domínio (descrições de produtos)
- **Diferença clara** vs modelo base
- **Sem overfitting** excessivo (1 época previne isso)
- **Generalização adequada** para títulos novos

**Resultado**: treino 10x mais rápido com qualidade muito similar!