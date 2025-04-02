# MLA-Self-Driving

Just testing. 

marked up verison of: [https://github.com/ambisinister/mla-experiments]

With CULANE dataset: [https://xingangpan.github.io/projects/CULane.html]
(about 10% used for training)

OG paper from: [https://arxiv.org/abs/1712.06080]

# Lane Detection with Multi-Head Latent Attention (MLA)

This project implements a lane detection neural network using a ResNet18 backbone combined with a custom Multi-Query Latent Attention (MLA) mechanism. The model takes in road images and predicts multiple lane points in normalized coordinates.
I wanted to utilize MLA for something other than NLP. Latent vectors may be good for future feature fusion. 

---

## Dataset Structure from CULANE [https://xingangpan.github.io/projects/CULane.html], used 10%

- Images: `.jpg` files (e.g., `0001.jpg`)
- Labels: `.lines.txt` files containing the coordinates of each lane in the corresponding image (e.g., `0001.lines.txt`)

Each line in a `.lines.txt` file represents a single lane, with space-separated x/y coordinates.

---

## Model Architecture

### 1. `LaneDataset`

Custom PyTorch `Dataset` class to load lane images and associated lane coordinate annotations. It:
- Loads images and corresponding `.lines.txt` files
- Normalizes lane coordinates relative to image width/height
- Pads lane and point data to a fixed shape: `[max_lanes, max_points, 2]`

```python
max_lanes = 5
max_points = 32
```

---

### 2. `MLA` (Multi-query Latent Attention)

Implements a multi-head attention module that projects input features to query, key, and value vectors and computes attention across them.

```python
MLA(input_dim=512, latent_dim=64, num_heads=8)
```

---

### 3. `LaneDetectionModel`

The main model consists of:
- A ResNet18 backbone (without the classification head)
- MLA attention mechanism
- Two fully connected layers for regression
- Final output reshaped to `[batch_size, max_lanes, max_points, 2]`

---

## Training

### Training Function

```python
train_model(model, dataloader, criterion, optimizer, num_epochs=10)
```

Performs standard model training using MSE loss.

---

### Evaluation Function

```python
evaluate_model(model, dataloader)
```

Computes:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R²) score

---

##

Update the `data_dir` variable with your dataset path:

```python
data_dir = "/path/to/your/data"
```

Then run the script:

```bash
python lane_detection.py
```

---

##  Example Output

During execution, the script will print:
- Dataset size
- Sample file names
- Loss per epoch
- Final evaluation metrics (MSE, MAE, R²)

---

## Dependencies

```bash
pip install torch torchvision scikit-learn numpy pillow
```

---

## other stuff 

- Padding is used to handle variable-length lane data.
- MLA is inspired by Transformer attention but simplified for spatial feature aggregation. 
- Batch size, learning rate, and max lanes/points can be tuned for better performance.
