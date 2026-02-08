# PointDiT flow:

**Image Flow:**

* **Input:** Input image -> `image` $(B, 1, 512, 512)$.
* **CNN Extraction:** 3-layer CNN extracts features from input image -> `img_feats` $(B, Dim, 64, 64)$.
* **Grid Creation:** Hardcoded Sin/Cos Fourier grid is generated -> `grid` $(B, 128, 64, 64)$.
* **Fusion:** Features and Grid are concatenated, flattened, and passed through Linear Layer -> `memory` $(B, 4096, Dim)$. (Note:  sequence length)


**Points Flow:**

* **Input:** Points initialized as noisy coordinates -> `x_t` $(B, N, 2)$.
* **Point Embedding:** Linear projection of coordinates -> `point_emb` $(B, N, Dim)$.
* **Time Embedding:** Sinusoidal embedding of timestep t -> `time_emb` $(B, 1, Dim)$.
* **Combination:** Element-wise addition of Point and Time embeddings -> `query` $(B, N, Dim)$.


**Combined Flow:**

* **Transformer:** `query` (points) attends to `memory` (image) via TransformerDecoder -> `out` $(B, N, Dim)$.
* **Prediction:** Output projected by Linear Head -> `pred_x0` $(B, N, 2)$ (Clean Coordinates).


# Losses flow:

All the losses are applied on the model final output:

* [USE] Chamfer -> (pred_x0, x_0) -> Compares predicted points to Ground Truth points.
* [USE] Sinkhorn -> (pred_x0, image) -> Checks if predicted points match the image density.
* [USE] Repulsion -> (pred_x0)-> Checks if predicted points are spacing themselves out (internal geometry).
* [NOT] Grid Density -> (pred_x0, image) -> Checks if predicted point density matches image density at multiple scales.
* [NOT] Spectral -> (pred_x0, x_0)-> Checks if the frequency "fingerprint" of the prediction matches the GT.
