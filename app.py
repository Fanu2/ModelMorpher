import streamlit as st
import torch
import tempfile
import os
from networks import Generator  # Replace with your actual architecture

ARCHITECTURES = {
    "Generator (CartoonGAN)": Generator
}

st.set_page_config(page_title="ModelMorpher Lite", page_icon="🧠", layout="centered")
st.title("🧠 ModelMorpher Lite: PyTorch → ONNX")
st.caption("Convert your .pth models into ONNX format for deployment.")

uploaded_file = st.file_uploader("📤 Upload your .pth model", type=["pth"])
arch_choice = st.selectbox("🧠 Choose model architecture", list(ARCHITECTURES.keys()))
input_shape = st.text_input("📐 Input shape (e.g. [1,3,256,256])", "[1,3,256,256]")

if uploaded_file and st.button("🔄 Convert to ONNX"):
    with st.spinner("Converting..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            pth_path = os.path.join(tmpdir, "model.pth")
            with open(pth_path, "wb") as f:
                f.write(uploaded_file.read())

            model_class = ARCHITECTURES[arch_choice]
            model = model_class()
            model.load_state_dict(torch.load(pth_path, map_location="cpu"))
            model.eval()

            dummy_input = torch.randn(*eval(input_shape))
            onnx_path = os.path.join(tmpdir, "model.onnx")
            torch.onnx.export(model, dummy_input, onnx_path,
                              input_names=["input"], output_names=["output"],
                              opset_version=11)

            st.success("✅ ONNX export complete!")
            st.download_button("📥 Download ONNX", open(onnx_path, "rb").read(), "model.onnx")
