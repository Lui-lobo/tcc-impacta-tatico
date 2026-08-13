import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.pipeline.quantizer import quantize_input, reference_interpreter

class ModelEvaluator:
    @staticmethod
    def _tflite_predictions(tflite_path, X_test):
        """Classes preditas pelo modelo QUANTIZADO, janela a janela.

        E este o modelo que roda no ESP32. A acuracia do Keras em float32 mede
        o modelo antes da quantizacao e nao descreve o sistema embarcado; as
        duas precisam aparecer lado a lado para que o custo da conversao fique
        visivel no relatorio.
        """
        interpreter = reference_interpreter(tflite_path)
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        scale, zero_point = inp['quantization']

        quantized = quantize_input(X_test.astype(np.float32), scale, zero_point)
        preds = np.zeros(len(X_test), dtype=int)
        for i in range(len(X_test)):
            interpreter.set_tensor(inp['index'], quantized[i].reshape(inp['shape']))
            interpreter.invoke()
            preds[i] = int(np.argmax(interpreter.get_tensor(out['index'])[0]))
        return preds

    @staticmethod
    def generate_metrics_report(model, history, X_test, y_test, tflite_path, output_report_path="relatorio_metricas.md", dataset_info=None):
        # Diretório base para os gráficos
        out_dir = os.path.dirname(output_report_path)
        if not out_dir: out_dir = "."
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. Inferência na base de teste
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # 2. Métricas quantitativas — float32 (Keras) e int8 (o que vai ao ESP32)
        acc = accuracy_score(y_test, y_pred)
        report_str = classification_report(y_test, y_pred, output_dict=False)
        cm = confusion_matrix(y_test, y_pred)

        y_pred_int8 = ModelEvaluator._tflite_predictions(tflite_path, X_test)
        acc_int8 = accuracy_score(y_test, y_pred_int8)
        report_int8_str = classification_report(y_test, y_pred_int8, output_dict=False)
        cm_int8 = confusion_matrix(y_test, y_pred_int8)

        # Quantas janelas mudaram de classe por causa da quantização.
        divergentes = int(np.sum(y_pred != y_pred_int8))
        delta_pp = (acc_int8 - acc) * 100

        # ========================================================
        # 3. Geração de Gráficos (Alta Qualidade para Relatório/TCC)
        # ========================================================
        # Gráfico A: Matrizes de Confusão, float32 e int8 lado a lado.
        # Mostrar as duas deixa visível se a quantização degradou alguma classe
        # em particular, o que a acurácia global sozinha esconderia.
        classes = ['Normal', 'Inner Race', 'Outer Race'] if len(cm) == 3 else [f'Classe {i}' for i in range(len(cm))]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, matriz, titulo, acuracia in (
            (axes[0], cm, 'Keras float32 (treino)', acc),
            (axes[1], cm_int8, 'TFLite int8 (embarcado no ESP32)', acc_int8),
        ):
            sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=classes, yticklabels=classes,
                        annot_kws={"size": 14}, cbar=False)
            ax.set_title(f'{titulo}\nAcurácia: {acuracia * 100:.2f}%', fontsize=13)
            ax.set_ylabel('Classe Real (Ground Truth)', fontsize=11)
            ax.set_xlabel('Classe Predita (Modelo)', fontsize=11)
        fig.suptitle('Matriz de Confusão do Modelo TinyML', fontsize=16)
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
                        
        # Seção opcional com a escala dos dados. O número de janelas
        # INDEPENDENTES é o que limita a validade estatística do resultado: as
        # janelas sobrepostas multiplicam a contagem sem acrescentar sinal novo.
        dataset_section = ""
        if dataset_info:
            independentes = dataset_info.get('janelas_independentes', 0)
            parecer = ("✅ Amostra adequada." if independentes >= 300 else
                       "⚠️ Amostra pequena: trate a acurácia como indicativa.")
            dataset_section = f"""
## 1.1 Escala do Conjunto de Dados

| Item | Valor |
|---|---|
| Arquivos `.mat` do CWRU | {dataset_info.get('arquivos', 'n/d')} |
| Sinal após decimação | {dataset_info.get('duracao_s', 0):.1f} s a {dataset_info.get('taxa_hz', 0)} Hz |
| Janelas de treino | {dataset_info.get('janelas_treino', 'n/d')} |
| Janelas de teste | {dataset_info.get('janelas_teste', 'n/d')} |
| **Janelas independentes** (sem sobreposição) | **{independentes}** |

{parecer} As janelas de treino e teste usam sobreposição de
{dataset_info.get('overlap', 0) * 100:.2f}%, o que multiplica a contagem sem criar
informação nova. O corte treino/teste é feito por trecho temporal contíguo,
antes do janelamento, com uma zona morta de uma janela inteira — nenhuma amostra
aparece dos dois lados.

---
"""

        # Formatação MD
        markdown_content = f"""# 📈 Relatório de Desempenho, Gráficos e Consumo (TinyML)

## 1. Visualização do Modelo (Gráficos exportados para o TCC)
As imagens abaixo foram geradas fisicamente em alta resolução (300 DPI) e salvas na pasta raiz de artefatos. Use-as para comprovar o aprendizado do seu modelo sem *overfitting*.

### 📊 Histórico de Treinamento (Curva de Aprendizado)
Este gráfico valida o aprendizado das extrações de frequência ao longo do tempo.
{f"![Curvas de Treinamento]({os.path.basename(hist_png)})" if hist_png else "*Gráfico indisponível.*"}

### 🎯 Matriz de Confusão Visual
Mapeamento termográfico exato de onde o modelo classificou corretamente ou sofreu Falsos Positivos/Negativos, antes e depois da quantização.
![Matriz de Confusão]({os.path.basename(cm_png)})

---
{dataset_section}
## 2. Métricas Analíticas

O modelo gravado no ESP32 é o **quantizado em int8**, não o Keras em float32. As
duas acurácias aparecem abaixo porque só a diferença entre elas revela o custo
real da conversão.

| Modelo | Acurácia | Papel |
|---|---|---|
| Keras float32 | `{acc * 100:.2f}%` | referência de projeto (não embarcável) |
| **TFLite int8** | **`{acc_int8 * 100:.2f}%`** | **é este o número a citar para o ESP32** |

* **Custo da quantização:** `{delta_pp:+.2f} pp` — {divergentes} de {len(y_test)} janelas mudaram de classe.
* Kernels de referência (`BUILTIN_REF`), que são os que o TensorFlow Lite Micro implementa.

> ⚠️ **Como ler esta acurácia.** A divisão treino/teste é temporal *dentro de cada
> arquivo*: os primeiros 80% de cada gravação vão para o treino e os últimos 20%
> para o teste. Não há vazamento de amostras — existe uma zona morta de uma janela
> inteira na fronteira — mas **toda janela de teste tem uma contraparte de treino
> da mesma gravação**: mesmo rolamento, mesmo defeito, mesma carga, mesma montagem.
> O número acima mede consistência dentro da condição de operação, não
> generalização para uma condição nova.
>
> Para a medida de generalização, execute `python tools/validacao_por_carga.py`,
> que treina em três níveis de carga e testa no quarto. Reporte os dois números no
> TCC: este é o limite superior, aquele é o realista.

**Precisão e Recall — TFLite int8 (modelo embarcado):**
```text
{report_int8_str}
```

**Precisão e Recall — Keras float32 (antes da quantização):**
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
