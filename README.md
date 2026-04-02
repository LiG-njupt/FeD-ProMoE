# FeD-ProMoE
Fed-ProMoE: Prototype-guided Federated MoE for Privacy-Preserving UAV Specific Emitter Identification (SEI) under Extreme Non-IID Data

📌 Project Overview

Specific Emitter Identification (SEI) provides a robust physical-layer solution for wireless device authentication via hardware-specific radio-frequency (RF) fingerprints, which is critical for low-altitude security and UAV wireless monitoring applications. However, in real-world distributed sensing scenarios, SEI data is inherently fragmented across multiple edge receivers, and cannot be centrally aggregated due to strict privacy, communication bandwidth, and storage constraints.

This challenge is further exacerbated under extreme heterogeneous non-IID data distributions, especially the pathological one-class-per-client setting (each edge receiver only observes signals from a single UAV emitter category). Conventional federated learning (FL) methods suffer from severe client drift, unstable decision boundaries, and weak global discriminability in this scenario.

To address these issues, we propose Fed-ProMoE, a prototype-guided federated mixture-of-experts (MoE) framework tailored for privacy-preserving distributed SEI. Our method enables effective cross-client knowledge transfer and global discriminability enhancement, without requiring raw RF signal sharing between edge clients or the cloud server.



Data partition: 8 clients with extreme one-class-per-client non-IID setting

📄 License
This project is released under the MIT License.

The specific code will be uploaded after the paper is published
