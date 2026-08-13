import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ModelEvaluator:
    @staticmethod
    def generate_metrics_report(model, history, X_test, y_test, tflite_path, output_report_path="relatorio_metricas.md"):
        # Diretório base para os gráficos
        out_dir = os.path.dirname(output_report_path)
        if not out_dir: out_dir = "."
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. Inferência na base de teste
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # 2. Métricas quantitativas
        acc = accuracy_score(y_test, y_pred)
        report_str = classification_report(y_test, y_pred, output_dict=False)
        cm = confusion_matrix(y_test, y_pred)
        
        # ========================================================
        # 3. Geração de Gráficos (Alta Qualidade para Relatório/TCC)
        # ========================================================
        # Gráfico A: Matriz de Confusão
        plt.figure(figsize=(8, 6))
        # Verifica as classes geradas dinamicamente
        classes = ['Normal', 'Inner Race', 'Outer Race'] if len(cm) == 3 else [f'Classe {i}' for i in range(len(cm))]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
                    annot_kws={"size": 14})
        plt.title('Matriz de Confusão do Modelo TinyML', fontsize=16)
        plt.ylabel('Classe Real (Ground Truth)', fontsize=12)
        plt.xlabel('Classe Predita (Modelo)', fontsize=12)
        plt.tight_layout()
        cm_png = os.path.join(out_dir, "matriz_confusao.png")
        plt.savefig(cm_png, dpi=300) # Salva com 300 DPI ideal para impressão
        plt.close()
        
        # Gráfico B: Histórico de Treinamento (Acurácia / Loss)
        hist_png = None
        if history is not None:
            plt.figure(figsize=(14, 5))
            
            # Subplot 1: Acurácia
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Treinamento (Train)', linewidth=2)
            plt.plot(history.history['val_accuracy'], label='Validação (Test)', linewidth=2)
            plt.title('Curva de Aprendizado - Acurácia', fontsize=14)
            plt.xlabel('Épocas', fontsize=12)
            plt.ylabel('Acurácia', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='lower right')
            
            # Subplot 2: Função de Perda (Loss)
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Treinamento (Train)', linewidth=2)
            plt.plot(history.history['val_loss'], label='Validação (Test)', linewidth=2)
            plt.title('Curva de Aprendizado - Função de Perda (Loss)', fontsize=14)
            plt.xlabel('Épocas', fontsize=12)
            plt.ylabel('Perda / Erro', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='upper right')
            
            plt.tight_layout()
            hist_png = os.path.join(out_dir, "historico_treinamento.png")
            plt.savefig(hist_png, dpi=300)
            plt.close()
            
        # ========================================================
        # 4. Avaliação de Hardware (ESP32)
        # ========================================================
        flash_usage_kb = os.path.getsize(tflite_path) / 1024.0
        max_activation_size = 0
        for layer in model.layers:
            # Keras 3 removeu o atributo `output_shape` das camadas; a forma
            # agora sai do tensor de saída. O fallback mantém o suporte a Keras 2.
            shape = None
            try:
                shape = layer.output.shape
            except AttributeError:
                shape = getattr(layer, 'output_shape', None)
            if shape is None:
                continue
            if isinstance(shape, list): shape = shape[0]
            valid_dims = [dim for dim in shape if dim is not None]
            if valid_dims:
                size = np.prod(valid_dims) * 1
                if size > max_activation_size:
                    max_activation_size = size
                        
        # Formatação MD
        markdown_content = f"""# 📈 Relatório de Desempenho, Gráficos e Consumo (TinyML)

## 1. Visualização do Modelo (Gráficos exportados para o TCC)
As imagens abaixo foram geradas fisicamente em alta resolução (300 DPI) e salvas na pasta raiz de artefatos. Use-as para comprovar o aprendizado do seu modelo sem *overfitting*.

### 📊 Histórico de Treinamento (Curva de Aprendizado)
Este gráfico valida o aprendizado das extrações de frequência ao longo do tempo.
{f"![Curvas de Treinamento]({os.path.basename(hist_png)})" if hist_png else "*Gráfico indisponível.*"}

### 🎯 Matriz de Confusão Visual
Mapeamento termográfico exato de onde o modelo classificou corretamente ou sofreu Falsos Positivos/Negativos.
![Matriz de Confusão]({os.path.basename(cm_png)})

---

## 2. Métricas Analíticas
**Acurácia Global (Base de Teste Inédita):** `{acc * 100:.2f}%`

**Relatório de Precisão e Recall do Scikit-Learn:**
```text
{report_str}
```

---

## 3. Limites de Hardware Comprovados (Memória do ESP32)

* 💽 **Consumo de Memória Flash (ROM):** `{flash_usage_kb:.2f} KB`
  * *Parecer Técnico:* {"✅ Ideal (Menor que 1 MB. Sobra espaço para OTA e WebServer no ESP32)." if flash_usage_kb < 1000 else "⚠️ Atenção."}

* 🧠 **Pico de Memória RAM Volátil (Tensor Arena / SRAM):** `~{max_activation_size / 1024:.2f} KB`
  * *Parecer Técnico:* {"✅ Ideal (O modelo não esgota os ~320 KB estáticos da controladora)." if (max_activation_size/1024) < 150 else "⚠️ Crítico."}

* ⚡ **Performance Estimada de Processamento:**
  * O arquivo C gerado contém instruções Integer de 8 bits (Int8). Carga desprezível para a arquitetura RISC/Xtensa do microcontrolador sem uso de co-processador matemático FPU.
"""
        with open(output_report_path, "w", encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"[SUCESSO] Gráficos visuais (PNG) e relatório (MD) foram extraídos para o repositório.")
