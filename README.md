Licensed Under Creative Commons No Derivatives Non-Commercial Open Source


AI

# Quantitative Calculous Engine (QCE)

A small, modular framework for steering generators (text, images, etc.) using
quantitative fields and gradient-based optimization in latent space.

## Files

- `qce_core.py`  
  Core abstractions: latent vectors, field specs, generic optimizer.

- `engine_text.py`  
  QCE for text: topic, length, sentiment fields (uses `G_text`, `Phi_text` stubs).

- `engine_image.py`  
  QCE for images: style, composition, symbolism fields (uses `G_img`, `Phi_img` stubs).

- `engine_multimodal.py`  
  Combined engine that optimizes a shared latent code for both text and image.

## Concept

1. Latent vector `z ∈ R^n`.
2. Generator `G(z) -> output` (text, image, etc.).
3. Feature extractor `Φ(output) -> features`.
4. Fields `E_k(features, ...) ≥ 0`.
5. Total field:
   

\[
   E_{\text{total}}(z) = \sum_k w_k E_k
   \]


6. Optimize `z` to minimize `E_total(z)` (Quantitative Calculous Breakdown in latent space).

## Usage (high level)

1. Implement real generators (`G_text`, `G_img`) and feature extractors (`Phi_text`, `Phi_img`).
2. Define field specs (which fields, targets, weights).
3. Call the appropriate `QCE_generate_*` function to get a steered output.

This repo is intentionally minimal and meant as a conceptual + experimental scaffold.
