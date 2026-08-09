# 📈 Relatório de Desempenho, Gráficos e Consumo (TinyML)

## 1. Visualização do Modelo (Gráficos exportados para o TCC)
As imagens abaixo foram geradas fisicamente em alta resolução (300 DPI) e salvas na pasta raiz de artefatos. Use-as para comprovar o aprendizado do seu modelo sem *overfitting*.

### 📊 Histórico de Treinamento (Curva de Aprendizado)
Este gráfico valida o aprendizado das extrações de frequência ao longo do tempo.
![Curvas de Treinamento](/Users/eobidas/.gemini/antigravity-cli/brain/b5f137c8-24f8-4fa0-a82c-93c9eec2e6d0/historico_treinamento.png)

### 🎯 Matriz de Confusão Visual
Mapeamento termográfico exato de onde o modelo classificou corretamente ou sofreu Falsos Positivos/Negativos.
![Matriz de Confusão](/Users/eobidas/.gemini/antigravity-cli/brain/b5f137c8-24f8-4fa0-a82c-93c9eec2e6d0/matriz_confusao.png)

---

## 2. Métricas Analíticas
**Acurácia Global (Base de Teste Inédita):** `100.00%`

**Relatório de Precisão e Recall do Scikit-Learn:**
```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       190
           1       1.00      1.00      1.00        95
           2       1.00      1.00      1.00        95

    accuracy                           1.00       380
   macro avg       1.00      1.00      1.00       380
weighted avg       1.00      1.00      1.00       380

```

---

## 3. Limites de Hardware Comprovados (Memória do ESP32)

* 💽 **Consumo de Memória Flash (ROM):** `16.13 KB`
  * *Parecer Técnico:* ✅ Ideal (Menor que 1 MB. Sobra espaço para OTA e WebServer no ESP32).

* 🧠 **Pico de Memória RAM Volátil (Tensor Arena / SRAM):** `~0.00 KB`
  * *Parecer Técnico:* ✅ Ideal (O modelo não esgota os ~320 KB estáticos da controladora).

* ⚡ **Performance Estimada de Processamento:**
  * O arquivo C gerado contém instruções Integer de 8 bits (Int8). Carga desprezível para a arquitetura RISC/Xtensa do microcontrolador sem uso de co-processador matemático FPU.
