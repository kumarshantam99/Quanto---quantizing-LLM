# Quanto---quantizing-LLM

Quantization is a technique to reduce the computational and memory costs of evaluating Deep Learning Models by representing their weights and activations with low-precision data types like 8-bit integer (int8) instead of the usual 32-bit floating point (float32).

Reducing the number of bits means the resulting model requires less memory storage, which is crucial for deploying Large Language Models on consumer devices. It also enables specific optimizations for lower bitwidth datatypes, such as int8 or float8 matrix multiplications on CUDA devices.

Many open-source libraries are available to quantize pytorch Deep Learning Models, each providing very powerful features, yet often restricted to specific model configurations and devices.

Also, although they are based on the same design principles, they are unfortunately often incompatible with one another.

The library provides us:

- Quantizations - int2, int4, int8 and float8 weights.
- supports int8 and float8 activations.
- quantized models can be placed on any device (including CUDA and MPS)
- Supports quantization aware Tuning
- Supports serialization and storage into pickle or safetensors.
