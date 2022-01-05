# CLIP-ONNX
It is a simple library to speed up CLIP inference up to 3x (K80 GPU)
## Usage
Install clip-onnx module and requirements first. Use this trick
```python3
!pip install git+https://github.com/Lednik7/CLIP-ONNX.git
```
## Example in 3 steps
0. Download CLIP image from repo
```python3
!wget -c -O CLIP.png https://github.com/openai/CLIP/blob/main/CLIP.png?raw=true
```
1. Load standard CLIP model, image, text on cpu
```python3
import clip
from PIL import Image

# onnx cannot work with cuda
model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
# batch first
image = preprocess(Image.open("CLIP.png")).unsqueeze(0) # [1, 3, 224, 224]
text = clip.tokenize(["a diagram", "a dog", "a cat"]) # [3, 77]
```
2. Create CLIP-ONNX object to convert model to onnx
```python3
from clip_onnx import clip_onnx, attention
clip.model.ResidualAttentionBlock.attention = attention

# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
onnx_model = clip_onnx(model, providers=["CPUExecutionProvider"]) # cpu mode
onnx_model.convert2onnx(image, text, verbose=True)
onnx_model.start_sessions()
```
3. Use for standard CLIP API. Batch inference
```python3
image_features = onnx_model.encode_image(image)
text_features = onnx_model.encode_text(text)

logits_per_image, logits_per_text = onnx_model(image, text)
probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.41456965 0.29270944 0.29272085]]
```
Enjoy the speed

## Examples
See [examples folder](https://github.com/Lednik7/CLIP-ONNX/tree/main/examples) for more details \
Some parts of the code were taken from the [post](https://twitter.com/apeoffire/status/1478493291008172038). Thank you [neverix](https://github.com/neverix) for this notebook.