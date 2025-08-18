# 🧠 ModelMorpher: PyTorch → ONNX → OpenVINO IR Converter

ModelMorpher is a Streamlit app that lets you upload `.pth` models, select architecture, and convert them into ONNX and OpenVINO IR formats for deployment.

## 🚀 Features
- Upload `.pth` PyTorch models
- Choose architecture (Generator, ResNet, UNet)
- Set input shape
- Export to ONNX and OpenVINO IR
- Download `.onnx`, `.xml`, and `.bin` files

## 🛠️ Setup

```bash

git clone https://github.com/your-username/ModelMorpher.git
cd ModelMorpher
pip install -r requirements.txt
streamlit run app.py
streamlit
torch
openvino-dev


ModelMorpher/
├── app.py
├── networks/
│   ├── __init__.py
│   ├── generator.py
│   ├── resnet.py
│   └── unet.py
├── requirements.txt
├── README.md
└── .gitignore
