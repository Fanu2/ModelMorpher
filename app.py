import streamlit as st
import torch
import os

st.set_page_config(page_title="🧠 Model Converter", layout="wide")

# --- Tab Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["📤 Load Model", "🔍 Preview", "🔁 Convert", "📥 Export"])

# --- Shared State ---
model = None
input_shape = None
onnx_path = None

# --- Tab 1: Load Model ---
with tab1:
    st.header("📤 Upload or Load Model")
    model_file = st.file_uploader("Upload .pth model", type=["pth"])
    input_shape_str = st.text_input("Input shape (e.g. 1,3,224,224)", "1,3,224,224")

    if model_file and input_shape_str:
        try:
            input_shape = tuple(map(int, input_shape_str.split(",")))
            model = torch.load(model_file, map_location="cpu")
            st.success("✅ Model loaded successfully.")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")

# --- Tab 2: Preview ---
with tab2:
    st.header("🔍 Model Preview")
    if model:
        st.write("Model type:", type(model))
        st.write("Input shape:", input_shape)
        st.code(str(model), language="python")
    else:
        st.warning("⚠️ No model loaded yet. Please go to Tab 1.")

# --- Tab 3: Convert ---
with tab3:
    st.header("🔁 Convert to ONNX")
    if model and input_shape:
        output_dir = st.text_input("📁 Output directory", "converted_models")
        output_name = st.text_input("📝 ONNX filename", "model.onnx")

        if st.button("💾 Convert to ONNX"):
            try:
                dummy_input = torch.randn(*input_shape)
                os.makedirs(output_dir, exist_ok=True)
                onnx_path = os.path.join(output_dir, output_name)

                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=11
                )

                st.success(f"✅ ONNX model saved to: `{onnx_path}`")
            except Exception as e:
                st.error(f"❌ Conversion failed: {e}")
    else:
        st.warning("⚠️ Model not ready. Please complete Tab 1.")

# --- Tab 4: Export ---
with tab4:
    st.header("📥 Download ONNX")
    if onnx_path and os.path.exists(onnx_path):
        with open(onnx_path, "rb") as f:
            st.download_button("📥 Download ONNX", data=f.read(), file_name=os.path.basename(onnx_path))
    else:
        st.info("ℹ️ No ONNX file available yet. Convert in Tab 3.")
