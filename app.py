import streamlit as st
import torch
import tempfile
import os
import shutil
from openvino.tools.mo import convert_model

# Optional: import your known architectures
from networks import Generator, ResNet, UNet  # Replace with your actual modules

ARCHITECTURES = {
    "Generator (CartoonGAN)": Generator,
    "ResNet": ResNet,
    "UNet": UNet
}

st.set_page_config(page_title="ModelMorpher", page_icon="🧠", layout="centered")
st.title("🧠 ModelMorpher: PyTorch → ONNX → OpenVINO IR")
st.caption("Convert your .pth models into deployable formats with ease.")

# Upload section
uploaded_file = st.file_uploader("📤 Upload your .pth model", type=["pth"])
arch_choice = st.selectbox("🧠 Choose model architecture", list(ARCHITECTURES.keys()))
input_shape = st.text_input("📐 Input shape (e.g. [1,3,256,256])", "[1,3,256,256]")

# Conversion trigger
if uploaded_file and st.button("🔄 Convert"):
    with st.spinner("Converting..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            pth_path = os.path.join(tmpdir, "model.pth")
            with open(pth_path, "wb") as f:
                f.write(uploaded_file.read())

            # Load model
            model_class = ARCHITECTURES[arch_choice]
            model = model_class()
            model.load_state_dict(torch.load(pth_path, map_location="cpu"))
            model.eval()

            # Export to ONNX
            dummy_input = torch.randn(*eval(input_shape))
            onnx_path = os.path.join(tmpdir, "model.onnx")
            torch.onnx.export(model, dummy_input, onnx_path,
                              input_names=["input"], output_names=["output"],
                              opset_version=11)

            # Convert to OpenVINO IR
            ir_output_dir = os.path.join(tmpdir, "ir_output")
            os.makedirs(ir_output_dir, exist_ok=True)
            ir_model = convert_model(onnx_path, output_dir=ir_output_dir)

            # Prepare downloads
            xml_path = os.path.join(ir_output_dir, "model.xml")
            bin_path = os.path.join(ir_output_dir, "model.bin")

            st.success("✅ Conversion complete!")
            st.download_button("📥 Download XML", open(xml_path, "rb").read(), "model.xml")
            st.download_button("📥 Download BIN", open(bin_path, "rb").read(), "model.bin")
            st.download_button("📥 Download ONNX", open(onnx_path, "rb").read(), "model.onnx")
