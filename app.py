import streamlit as st
import torch
import os

# 🔧 ONNX Conversion Function
def convert_to_onnx(model, input_shape):
    st.subheader("🔁 Convert to ONNX")

    output_dir = st.text_input("📁 ONNX output directory", "converted_models")
    output_name = st.text_input("📝 ONNX filename", "model.onnx")

    quantize = st.checkbox("⚡ Quantize ONNX (experimental)")
    fallback = st.checkbox("🛟 Fallback to TorchScript if ONNX fails")

    if st.button("💾 Convert to ONNX"):
        st.write("🔧 Starting ONNX conversion...")
        try:
            dummy_input = torch.randn(*input_shape)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_name)

            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=11
            )

            if quantize:
                try:
                    from onnxruntime.quantization import quantize_dynamic, QuantType
                    quantized_path = output_path.replace(".onnx", "_quantized.onnx")
                    quantize_dynamic(output_path, quantized_path, weight_type=QuantType.QInt8)
                    output_path = quantized_path
                    st.success("⚡ Quantization successful")
                except Exception as qe:
                    st.warning(f"⚠️ Quantization failed: {qe}")

            st.success(f"✅ ONNX model saved to: `{output_path}`")

            with open(output_path, "rb") as f:
                st.download_button("📥 Download ONNX", data=f.read(), file_name=os.path.basename(output_path))

            return output_path

        except Exception as e:
            st.error(f"❌ ONNX conversion failed: {e}")
            if fallback:
                try:
                    ts_path = os.path.join(output_dir, output_name.replace(".onnx", ".pt"))
                    scripted_model = torch.jit.trace(model, dummy_input)
                    scripted_model.save(ts_path)
                    st.success(f"🛟 Fallback TorchScript saved to: `{ts_path}`")
                    with open(ts_path, "rb") as f:
                        st.download_button("📥 Download TorchScript", data=f.read(), file_name=os.path.basename(ts_path))
                    return ts_path
                except Exception as te:
                    st.error(f"❌ TorchScript fallback also failed: {te}")
            return None
    return None

# 📂 Model Loader
def load_model():
    st.sidebar.header("📂 Load Model")
    model_source = st.sidebar.radio("Choose model source", ["Pretrained", "Upload .pth", "Select from directory"])
    input_dims = st.sidebar.text_input("📐 Input shape (e.g. 1,3,224,224)", "1,3,224,224")

    try:
        input_shape = tuple(map(int, input_dims.strip().split(",")))
    except:
        st.sidebar.error("❌ Invalid input shape format.")
        return None, None

    def get_architecture(arch_name):
        if arch_name == "vgg19":
            from torchvision.models import vgg19
            return vgg19()
        elif arch_name == "resnet18":
            from torchvision.models import resnet18
            return resnet18()
        elif arch_name == "mobilenet_v2":
            from torchvision.models import mobilenet_v2
            return mobilenet_v2()
        else:
            st.error("Unknown architecture.")
            return None

    if model_source == "Pretrained":
        model_name = st.sidebar.selectbox("Select model", ["resnet18", "mobilenet_v2", "vgg19"])
        try:
            if model_name == "resnet18":
                from torchvision.models import resnet18, ResNet18_Weights
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
            elif model_name == "mobilenet_v2":
                from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
                model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            elif model_name == "vgg19":
                from torchvision.models import vgg19, VGG19_Weights
                model = vgg19(weights=VGG19_Weights.DEFAULT)
            else:
                st.error("Unknown model selected.")
                return None, None
            st.success(f"✅ Loaded {model_name}")
            return model.eval(), input_shape
        except Exception as e:
            st.error(f"❌ Failed to load pretrained model: {e}")
            return None, None

    elif model_source == "Upload .pth":
        uploaded_file = st.sidebar.file_uploader("Upload .pth file", type=["pth"])
        arch = st.sidebar.selectbox("Model architecture", ["vgg19", "resnet18", "mobilenet_v2"])
        if uploaded_file:
            try:
                model = get_architecture(arch)
                state_dict = torch.load(uploaded_file, map_location="cpu")
                model.load_state_dict(state_dict)
                st.success(f"✅ Loaded weights into {arch}")
                return model.eval(), input_shape
            except Exception as e:
                st.error(f"❌ Failed to load weights: {e}")
        return None, None

    else:  # Select from directory
        model_dir = st.sidebar.text_input("📁 Model directory", "models")
        arch = st.sidebar.selectbox("Model architecture", ["vgg19", "resnet18", "mobilenet_v2"])
        try:
            files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
        except FileNotFoundError:
            st.sidebar.warning("⚠️ Directory not found.")
            return None, None

        if not files:
            st.sidebar.info("📂 No .pth files found in directory.")
            return None, None

        selected_file = st.sidebar.selectbox("Select model file", files)
        if selected_file:
            try:
                model_path = os.path.join(model_dir, selected_file)
                model = get_architecture(arch)
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict)
                st.success(f"✅ Loaded weights into {arch}")
                return model.eval(), input_shape
            except Exception as e:
                st.error(f"❌ Failed to load weights: {e}")
        return None, None

# 🧠 Main App
st.title("🧠 Model Morpher: ONNX Converter")

model, input_shape = load_model()

if model and input_shape:
    convert_to_onnx(model, input_shape)
else:
    st.info("📌 Please load a model and define input shape to begin.")
