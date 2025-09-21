# Diagnóstico: Fine-tuning não está funcionando

## Problema atual
Resultados idênticos entre modelo base e adapter no Streamlit.

## Alterações feitas no Streamlit

1. **Adicionado logs de debug**: O app agora mostra se o adapter está carregado corretamente
2. **Verificação PEFT**: Confirma se PEFT está ativo durante a geração
3. **Status do modelo**: Mostra se está em modo eval

## Como diagnosticar

### 1. Execute o Streamlit e observe os logs:
```
✓ Adapter carregado de: [caminho]
✓ PEFT config detectado: ['default']
✓ Adapters ativos: ['default']
🔧 PEFT ativo com 1 adapter(s) | ✓ Modelo em modo eval
```

Se não aparecer "PEFT ativo", o adapter não está sendo aplicado.

### 2. Teste com parâmetros determinísticos:
- `do_sample = False` (desabilitar)
- `temperature = 0.1` (baixo)
- `top_p = 1.0` (alto)

Se ainda assim os resultados forem idênticos, o problema não é aleatoriedade.

### 3. Teste títulos específicos do treino:
Use títulos que você SABE que estavam no dataset de treino para ver se há overfitting.

## Possíveis causas e soluções

### Causa 1: Dataset muito pequeno (200 registros → 71 de treino)
**Diagnóstico**: 71 exemplos podem não ser suficientes para TinyLlama aprender padrões significativos.

**Solução**: 
- Aumente `RECORDS_TO_FINETUNNING` para pelo menos 2000-5000
- No notebook: execute apenas as células de treino (pule baseline se quiser ir rápido)

### Causa 2: Learning rate muito alto/baixo
**Diagnóstico**: LR=2e-4 pode ser alto demais para um dataset pequeno.

**Solução**:
```python
# No notebook, célula de parâmetros:
OSS_LR = 1e-4  # ou 5e-5 (mais conservador)
```

### Causa 3: Épocas insuficientes
**Diagnóstico**: 2 épocas podem não ser suficientes.

**Solução**:
```python
OSS_EPOCHS = 3  # ou 4
```

### Causa 4: LoRA rank muito baixo
**Diagnóstico**: r=16 pode ser insuficiente para capturar mudanças.

**Solução**:
```python
LORA_R = 32  # ou 64
LORA_ALPHA = 32  # mantém proporção 1:1
```

### Causa 5: Modelo não convergiu
**Diagnóstico**: Loss não diminuiu durante treino.

**Solução**: Verificar os logs de treino no Colab:
- Loss deve diminuir gradualmente
- Eval loss deve acompanhar train loss
- Se loss não muda, aumentar LR ou verificar dados

## Teste rápido recomendado

1. **Aumente dataset**:
```python
RECORDS_TO_FINETUNNING = 2000  # no lugar de 200
```

2. **Ajuste hiperparâmetros**:
```python
OSS_LR = 1e-4
OSS_EPOCHS = 3
LORA_R = 32
LORA_ALPHA = 32
```

3. **Execute treino** e observe se loss diminui consistentemente

4. **Teste no Streamlit** com:
   - do_sample = False
   - Títulos específicos que estavam no treino
   - Observe os logs de debug

## Verificação final

### Teste A/B direto:
```python
# No final do notebook, adicione célula de teste:
test_title = "Smartphone with 5G connectivity"
prompt = f"[SYSTEM]\nYou are an assistant that writes detailed product descriptions from a given title.\n[USER]\nTitle: {test_title}\n[ASSISTANT]\n"

# Base
base_out = generate_batch([prompt], do_sample=False)[0]
print("BASE:", base_out)

# Fine-tuned (usando model após treino)
ft_out = generate_batch([prompt], do_sample=False)[0]  
print("FINE-TUNED:", ft_out)
```

Se forem idênticos, o treino realmente não funcionou.

## Quando considerar bem-sucedido

- Logs mostram "PEFT ativo" no Streamlit
- Resultados diferentes entre base e adapter (mesmo com seed fixo)
- Adapter gera descrições mais detalhadas/específicas
- Loss diminuiu durante treino no Colab

## Última alternativa: teste com merged model

Se adapter continuar problemático, use o modelo merged:
1. Execute a célula de merge no Colab
2. No Streamlit, escolha "Fine-tuned (merged)"
3. Compare base vs merged

O merged elimina variáveis da camada PEFT.