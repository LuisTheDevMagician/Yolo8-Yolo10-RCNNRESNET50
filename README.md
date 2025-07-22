
# Face Detection with Deep Learning Models

Este repositório contém a implementação, comparação e avaliação de três modelos de detecção de rostos em imagens com variações extremas de escala, pose, iluminação e oclusão. Os modelos analisados foram:

- **YOLOv8**
- **YOLOv10**
- **Faster R-CNN com ResNet-50**

## 📁 Estrutura do Projeto

```
.
├── yolo8.ipynb
├── yolo10.ipynb
├── rcnnResnet50.ipynb
├── /results
│   ├── all_results_yolo8.png
│   ├── all_results_yolo10.png
│   ├── confusion_matrix_yolo8.png
│   ├── confusion_matrix_yolo10.png
│   ├── confusion_matrix_rcnn.png
│   ├── score_rcnn.png
```

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8+
- PyTorch
- OpenCV
- Matplotlib
- Ultralytics (para YOLOv8 e YOLOv10)


### Execução

Execute cada notebook separadamente no Jupyter ou VS Code:

```bash
jupyter notebook yolo8.ipynb
jupyter notebook yolo10.ipynb
jupyter notebook rcnnResnet50.ipynb
```

---

# 📄 Explicação Detalhada dos Modelos de Detecção de Faces

## 🔍 YOLOv8 - Detecção com Ultralytics

### 1. Instalação de Dependências
```python
!pip install -q ultralytics kaggle
```
- `ultralytics`: biblioteca oficial do YOLOv8.
- `kaggle`: usada para baixar datasets da plataforma Kaggle.

### 2. Upload do kaggle.json
```python
from google.colab import files
files.upload()
```
- Permite autenticar com a API do Kaggle via chave `kaggle.json`.

### 3. Configuração de Acesso ao Kaggle
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
- Cria a pasta `.kaggle` e protege o arquivo com as permissões corretas.

### 4. Download e Extração do Dataset WIDER FACE
```bash
!kaggle datasets download -d lylmsc/wider-face-for-yolo-training
!unzip -q wider-face-for-yolo-training.zip -d wider_face
```
- Realiza o download e extrai o dataset adaptado para treinamento com YOLO.

### 5. Carregamento do Modelo YOLOv8
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```
- Utiliza o modelo "nano" (`yolov8n`), mais leve e rápido.

### 6. Treinamento do Modelo
```python
model.train(data="wider_face/data.yaml", epochs=10)
```
- Inicia o treinamento com base nas anotações do arquivo `data.yaml`.

### 7. Avaliação de Desempenho
```python
metrics = model.val()
```
- Realiza a validação e gera métricas como mAP, Precisão e Recall.

---

## 🔍 Faster R-CNN com ResNet50

### 1. Instalação de Bibliotecas
```python
!pip install -q torch torchvision albumentations kaggle
```
- `torch`, `torchvision`: base do PyTorch.
- `albumentations`: augmentação de dados.
- `kaggle`: para download do dataset.

### 2. Autenticação e Download do Dataset
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d lylmsc/wider-face-for-yolo-training
!unzip -q wider-face-for-yolo-training.zip -d wider_face
```

### 3. Augmentação e Dataset Customizado
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
```
- Realiza `Resize`, `Normalize`, conversão para tensor.
- Cria classe `Dataset` que lê imagens e arquivos `.txt` com bounding boxes.

### 4. Criação do Modelo Faster R-CNN
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
model = fasterrcnn_resnet50_fpn(pretrained=True)
```
- Utiliza modelo com backbone ResNet50 e rede FPN para detecção multiescala.

### 5. Treinamento do Modelo
- Otimizador: `torch.optim.SGD` ou `Adam`.
- Loop com `loss.backward()` e `optimizer.step()`.

### 6. Avaliação do Desempenho
- Gera matriz de confusão com TP, FP, FN e TN.
- Calcula:
  - `Precisão = TP / (TP + FP)`
  - `Recall = TP / (TP + FN)`
  - `F1 Score = 2 * (Precisão * Recall) / (Precisão + Recall)`

### 7. Visualização
- Plota matriz de confusão com `seaborn`.
- Gráfico de barras com métricas.

---

## 📊 Resultados e Comparações

### YOLOv8

- **mAP50**: ~0.59  
- **mAP50-95**: ~0.32  
- **Precisão**: ↑ crescente até ~0.82  
- **Recall**: até ~0.52  
- **Matriz de confusão**:
  ![YOLOv8 Confusion Matrix](./results/confusion_matrix_yolo8.png)

---

### YOLOv10

- **mAP50**: ~0.56  
- **mAP50-95**: ~0.32  
- **Precisão**: até ~0.81  
- **Recall**: até ~0.52  
- **Matriz de confusão**:
  ![YOLOv10 Confusion Matrix](./results/confusion_matrix_yolo10.png)

---

### Faster R-CNN (ResNet-50)

- **Precisão**: 0.14  
- **Recall**: 0.01  
- **F1-Score**: 0.01  
- **Matriz de confusão**:
  ![Faster R-CNN Confusion Matrix](./results/confusion_matrix_rcnn.png)

- **Gráfico de métricas**:
  ![Faster R-CNN Scores](./results/score_rcnn.png)

---

## 📌 Conclusões

- Os modelos **YOLOv8** e **YOLOv10** apresentaram resultados semelhantes, com leve vantagem para o YOLOv8 em precisão.
- O modelo **Faster R-CNN** com backbone ResNet-50 teve desempenho significativamente inferior neste dataset específico, com baixo recall e F1-score, indicando falhas na generalização.
- Todos os modelos foram testados com imagens da base **WIDER FACE**, com desafios extremos de detecção.

---

## 🧪 Tecnologias Utilizadas

- [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics)
- [YOLOv10](https://github.com/WongKinYiu/yolov10)
- [Faster R-CNN - torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
- Python, PyTorch, OpenCV, Matplotlib

---

## 👨‍💻 Autor

**Luis Eduardo**  
Estudante de Análise e Desenvolvimento de Sistemas - IFPI  
Proficiente em Next.js, Node.js e bancos de dados relacionais  
GitHub: [@LuisTheDevMagician](https://github.com/LuisTheDevMagician)

---

