# FeD-ProMoE
Fed-ProMoE: Prototype-guided Federated MoE for Privacy-Preserving UAV Specific Emitter Identification (SEI) under Extreme Non-IID Data

📌 Project Overview
Specific Emitter Identification (SEI) provides a robust physical-layer solution for wireless device authentication via hardware-specific radio-frequency (RF) fingerprints, which is critical for low-altitude security and UAV wireless monitoring applications. However, in real-world distributed sensing scenarios, SEI data is inherently fragmented across multiple edge receivers, and cannot be centrally aggregated due to strict privacy, communication bandwidth, and storage constraints.
This challenge is further exacerbated under extreme heterogeneous non-IID data distributions, especially the pathological one-class-per-client setting (each edge receiver only observes signals from a single UAV emitter category). Conventional federated learning (FL) methods suffer from severe client drift, unstable decision boundaries, and weak global discriminability in this scenario.
To address these issues, we propose Fed-ProMoE, a prototype-guided federated mixture-of-experts (MoE) framework tailored for privacy-preserving distributed SEI. Our method enables effective cross-client knowledge transfer and global discriminability enhancement, without requiring raw RF signal sharing between edge clients or the cloud server.

✨ Key Features & Core Innovations
Privacy-Enhanced Proxy Sample Construction
Each client constructs compact representative proxy samples from local feature embeddings, and applies Mixup augmentation to generate privacy-enhanced mixed proxies. This avoids raw data leakage while providing sufficient information for cloud-side collaborative learning.
Cloud-Side Expert Refinement with Collaborative Mutual Distillation
We maintain class-specific experts on the cloud, refined via cross-entropy loss and triplet loss for enhanced class discriminability. A collaborative mutual distillation mechanism is further designed to enable complementary knowledge transfer between experts, smoothing cross-class decision boundaries and mitigating overconfidence on fragmented data.
Prototype-Guided Adaptive Expert Routing
A learnable gate network is trained to produce adaptive routing weights for expert outputs, realizing dynamic expert fusion for heterogeneous RF signals, which outperforms static ensemble aggregation in extreme non-IID settings.
Confidence-Aware Hybrid Inference
A local-first inference strategy is designed: high-confidence samples are processed locally on the edge to reduce communication overhead, while low-confidence samples are offloaded to the cloud for hybrid inference with fused local-cloud outputs.
Component	Version
Python	3.13.5+
PyTorch	2.8.0+
CUDA	Compatible with NVIDIA GeForce GTX 5060 GPU
Other Dependencies	numpy, scipy, librosa, matplotlib

📊 Dataset
We validate our method on the DroneRFa dataset——DroneRFa: A large-scale dataset of drone radio frequency signals for detecting low-altitude drones, which contains I/Q RF signals collected from 8 outdoor flying UAVs.
Sampling rate: 100 MS/s
Center frequency: 2440 MHz
Input format: STFT spectrogram (244 × 244 single-channel)
Data partition: 8 clients with extreme one-class-per-client non-IID setting

📄 License
This project is released under the MIT License.
