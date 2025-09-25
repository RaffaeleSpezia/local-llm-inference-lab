

# MI50 LLM Inference — Complete Guide (ROCm)

*Author: Raffaele Spezia — September 25, 2025*  
*License: same as the repository license (e.g. MIT/Apache-2.0)*

> **Goal:** Deploy an **AMD Instinct MI50** GPU for local LLM inference, covering ROCm installation, PyTorch/Transformers setup, sample code, performance tuning and troubleshooting.

---

## Table of Contents

- [Overview](#overview)

- [Requirements](#requirements)

- [BIOS and Hardware Setup](#bios-and-hardware-setup)

- [ROCm Installation (driver + toolchain)](#rocm-installation-driver--toolchain)

- [Python Environment (PyTorch ROCm + Transformers)](#python-environment-pytorch-rocm--transformers)

- [Example 1 – Inference with Transformers on MI50](#example1--inference-with-transformers-on-mi50)

- [Example 2 – llama.cpp using HIP/ROCm](#example2--llamacpp-using-hiprocm)

- [Optimizations](#optimizations)

- [Monitoring](#monitoring)

- [Troubleshooting](#troubleshooting)

- [Quick FAQ](#quick-faq)

- [License](#license)

---

## Overview

The **AMD Instinct MI50** (Vega 20, 16–32 GB HBM2) remains attractive for **local LLM inference** thanks to its large VRAM at low second-hand cost.  
With **ROCm**, you can run PyTorch and Hugging Face Transformers almost like on CUDA; alternatively you can use **llama.cpp** via the **HIP backend**.

> **Outcome:** a stable and reasonably fast setup for models in the 7B–13B range (fp16 or quantized) with ready-to-use examples.

---

## Requirements

**Hardware**

- AMD Instinct **MI50** (preferably 32 GB HBM2)

- PSU ≥ 800 W (for multi-GPU servers), **2×8-pin** PCIe power

- High airflow (many MI50s are passive cards — you need strong chassis fans)

- PCIe x16 slot and motherboard with **UEFI + Above 4 G Decoding** support

**Software**

- Linux (**Ubuntu 22.04 LTS** recommended)

- ROCm **compatible with gfx906** (typically 6.2–6.3; newer releases deprecate MI50)

- Python 3.10/3.11 and `venv` or Conda

---

## BIOS and Hardware Setup

1. **UEFI mode only** – disable CSM/Legacy.

2. **Above 4 G Decoding: ON** – required for GPUs with large VRAM.

3. **Resizable BAR** – optional; enable if available.

4. **Power** – connect **both 8-pin PCIe connectors** directly to strong PSU rails.

5. **Cooling** – provide strong airflow through the MI50’s passive heatsink.

Check detection from Linux:

```bash
lspci | grep -Ei "Vega 20|MI50|Radeon Pro VII"
```

---

## ROCm Installation (driver + toolchain)

> Use a **ROCm release known to support gfx906** (MI50).

1. Add AMD repository & installer (example, adjust version):

```bash
wget https://repo.radeon.com/amdgpu-install/22.40.3/ubuntu/jammy/amdgpu-install_22.40.3_all.deb
sudo dpkg -i amdgpu-install_22.40.3_all.deb
```

2. Install ROCm components + kernel driver:

```bash
sudo amdgpu-install \
  --usecase=rocm,lrt,opencl,openclsdk,hip,hiplibsdk,dkms,mllib \
  --vulkan=amdvlk
```

3. Reboot and verify:

```bash
rocminfo      # MI50 should appear as agent gfx906
rocm-smi      # temperature, VRAM, power, etc.
lsmod | grep amdgpu
```

4. User permissions:

```bash
sudo usermod -a -G render,video $USER
# logout/login or reboot
```

> If you must use a newer ROCm release and the MI50 is not detected, you can try  
> `export HSA_OVERRIDE_GFX_VERSION=9.0.6` as a temporary workaround.  
> Prefer a natively supported ROCm version whenever possible.

---

## Python Environment (PyTorch ROCm + Transformers)

Create a virtual environment:

```bash
python3 -m venv ~/envs/rocm-llm
source ~/envs/rocm-llm/bin/activate
```

Install **PyTorch ROCm** (example for ROCm 6.3):

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

Install Hugging Face libraries:

```bash
pip install transformers accelerate scipy huggingface_hub
```

Smoke-test:

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))
```

---

## Example 1 – Inference with Transformers on MI50

```python
# run_llm_rocm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "User: Hi, how are you?\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Run:

```bash
python run_llm_rocm.py
```

---

## Example 2 – llama.cpp using HIP/ROCm

Build:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_HIP=ON
cmake --build build -j$(nproc)
```

Run (with a GGUF 4-bit model):

```bash
./build/bin/llama-cli \
  -m models/llama-2-7b.Q4_0.gguf \
  -p "Hello, MI50!" \
  -n 128 \
  -ngl 100     # offload all layers to GPU (if VRAM allows)
```

---

## Optimizations

- **Precision:** Always prefer **fp16**; for large models use 8- or 4-bit quantization (GGUF, GPTQ).

- **Throughput:** For multi-user serving, check **vLLM** forks with gfx906 support for dynamic batching.

- **Multi-GPU:** Hugging Face `Accelerate device_map="auto"` can shard layers across multiple MI50s.

- **Memory management:** Batch size 1 for interactive chat; use quantization for long contexts.

- **Stability:** Avoid killing processes abruptly (to reduce ROCm “reset bug”).  
  If you experience rare freezes during heavy DMA transfers:
  
  ```bash
  export HSA_ENABLE_SDMA=0
  ```

---

## Monitoring

```bash
watch -n 1 rocm-smi
```

shows VRAM%, GPU%, temperature and power while generating tokens.

In PyTorch:

```python
torch.cuda.reset_peak_memory_stats()
# run inference
print(torch.cuda.max_memory_allocated() / 1024**3, "GB")
```

---

## Troubleshooting

**GPU not visible in `rocminfo` / PyTorch**

- BIOS: ensure **UEFI only** and **Above 4 G Decoding ON**.

- Driver: reinstall a ROCm version supporting **gfx906**.

- Permissions: add user to `render,video` and re-login.

- PCIe: use a proper x16 slot (avoid cheap risers).

**Segmentation fault / crash when VRAM is high**

- Likely out-of-memory or fragmentation. Reduce model size or quantize.

- Reboot and retry.

- Use `device_map="auto"` for partial CPU offload.

- llama.cpp with 4-bit models is often safer.

**Slow generation**

- Confirm tensors are on `cuda` (ROCm) not CPU.

- Avoid CUDA-only quantization libs (e.g. bitsandbytes).

- Warm-up: first tokens are slower due to ROCm kernel JIT.

**Freeze / GPU reset**

- Always shut down processes gracefully.

- Optionally set `HSA_ENABLE_SDMA=0` if freezes occur.

- Check power supply and temperatures.

---

## Quick FAQ

**Docker?** Yes, pass `/dev/dri/renderD*` to the container and use the same ROCm stack as host. Bare-metal remains simpler and faster.

**BitsAndBytes on AMD?** No — currently CUDA-only. Use pre-quantized GGUF/GPTQ models or llama.cpp.

**Two GPUs (e.g. MI50 + another)?** Yes. With `Accelerate device_map="auto"` you can shard a model across GPUs or run multiple inference servers.

**Recommended models for 16 GB VRAM?**  
7B fp16 or 13B int4. With 32 GB you can handle 13B fp16/8-bit and longer context windows.

---

## License



### Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

### © 2025 Raffaele Spezia.

---

*This guide condenses hands-on experience in configuring the AMD Instinct MI50 for modern LLM inference. Follow the steps carefully to achieve a stable and efficient setup on your own server.*
