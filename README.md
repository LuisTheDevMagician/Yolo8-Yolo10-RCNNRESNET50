
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
  <img width="3000" height="2250" alt="Image" src="https://github.com/user-attachments/assets/49fd1e11-47b9-4e58-9ff2-95a07011317a" />
- **Todos os Resultados**:
  <img width="2400" height="1200" alt="Image" src="https://github.com/user-attachments/assets/2fe4c67e-fbbe-4a44-a404-4735c9bb3a3a" />

  - **Imagem Validção**:
 ![validation-1](https://github.com/user-attachments/assets/14f96ea4-146a-4046-8216-efc0433908d0)

  - **Imagem Predição**:
  ![prediction-1](https://github.com/user-attachments/assets/882c35da-151e-4aa3-ab09-d373968acf74)


---

### YOLOv10

- **mAP50**: ~0.56  
- **mAP50-95**: ~0.32  
- **Precisão**: até ~0.81  
- **Recall**: até ~0.52  
- **Matriz de confusão**:
<img width="3000" height="2250" alt="Image" src="https://github.com/user-attachments/assets/48428b15-92f2-483c-9c68-917ea6e0e0e1" />
- **Todos os Resultados**:
<img width="2400" height="1200" alt="Image" src="https://github.com/user-attachments/assets/b9a95d64-8be0-4d6a-ac60-492b495bd1c3" />

- **Imagem Validção**:
![cc3106d1-d428-42bd-85c0-464408f01110](https://github.com/user-attachments/assets/0c43ea31-3f52-43cb-982d-dcc876f23e71)


- **Imagem Predição**:
![625a2850-4e4d-4e33-8dbf-21404103fcea](https://github.com/user-attachments/assets/7fd4e003-9f7f-4400-bc1f-9ed74a213c6f)

---

### Faster R-CNN (ResNet-50)

- **Precisão**: 0.14  
- **Recall**: 0.01  
- **F1-Score**: 0.01  
- **Matriz de confusão**:
<img width="528" height="470" alt="Image" src="https://github.com/user-attachments/assets/bf4f7381-205f-4ad2-b7bc-34bad0283717" />

- **Gráfico de métricas**:
<img width="590" height="390" alt="Image" src="https://github.com/user-attachments/assets/430a86be-5f62-493d-b4ee-1d3898645828" />

- **Imagem Predição**:
<img width="636" height="658" alt="prediction-2" src="https://github.com/user-attachments/assets/c412427c-b939-4724-8681-92402ffbc0f2" />



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

