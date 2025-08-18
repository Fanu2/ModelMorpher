import streamlit as st
import torch
import os
from networks import Generator  # Your actual architecture

st.set_page_config(page_title="🧠 Local ModelMorpher", layout="centered")
st.title("🧠 Convert PyTorch to ONNX for Hugging Face")

model_path = "models/vgg19-dcbb9e9d.pth"
input_shape = st.text_input("📐 Input shape", "[1,3,256,256]", key="shape")

if st.button("🔄 Convert to ONNX"):
    with st.spinner("Converting..."):
        model = Generator()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        dummy_input = torch.randn(*eval(input_shape))
        torch.onnx.export(model, dummy_input, "model.onnx",
                          input_names=["input"], output_names=["output"],
                          opset_version=11)

        st.success("✅ Conversion complete!")
        st.download_button("📥 Download ONNX", open("model.onnx", "rb").read(), "model.onnx")
