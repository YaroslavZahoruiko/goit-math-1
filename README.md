# Homework 1: Linear Algebra Applications

This repository contains four tasks demonstrating practical applications of linear algebra using NumPy.

## Requirements

- Python 3.x
- NumPy

**Note:** NumPy is pre-installed in Google Colab. For local setup, install with:
```bash
pip install numpy
```

## Tasks Overview

### Task 1: Image Modification (`task1_image_modification.ipynb`)

**Objective:** Implement image processing operations using matrix operations.

**Functions:**
- `change_contrast(M, scalar)`: Adjusts image contrast by multiplying the image matrix with a scalar
  - `scalar < 1`: Makes image darker
  - `scalar > 1`: Makes image brighter
  
- `change_brightness(M, scalar)`: Adjusts image brightness by adding a scalar to all pixels
  - `scalar > 0`: Makes image brighter
  - `scalar < 0`: Makes image darker
  - Values are clipped to [0, 255] range

- `blending(M, E, alpha)`: Blends two images using alpha blending
  - Formula: `alpha * M + (1 - alpha) * E`
  - `alpha = 0`: Result is image E
  - `alpha = 1`: Result is image M
  - `alpha = 0.5`: Equal mix of both images

**Key Concepts:**
- Matrix-scalar multiplication
- Matrix-scalar addition
- Linear combination of matrices
- Value clipping for valid pixel ranges

---

### Task 2: Best Matches Movie (`task2_best matches_movie.ipynb`)

**Objective:** Find the best movie recommendation for a user using cosine similarity.

**Approach:**
1. **Vector-based**: Computes cosine similarity between user preferences and each movie individually
2. **Matrix-based**: Computes cosine similarity for all movies at once using matrix operations

**Cosine Similarity Formula:**
```
cosine_similarity(u, v) = dot(u, v) / (norm(u) * norm(v))
```

**Key Concepts:**
- Dot product
- Vector normalization
- Cosine similarity (measures angle between vectors, range: -1 to 1)
- Matrix operations for batch processing
- Broadcasting with `keepdims=True` for proper normalization

**Result:** Returns the movie with the highest cosine similarity score (closest to 1 = most similar preferences).

---

### Task 3: Drones Factory (`task3_drons_factory.ipynb`)

**Objective:** Solve a system of linear equations to determine production quantities.

**Problem:** A factory produces three types of drones (Scout, Kamikaze, Cargo) using different amounts of materials. Given total available materials, find how many of each drone type can be produced.

**System of Equations:**
```
A @ x = b
```

Where:
- `A`: Matrix representing material requirements per drone type
- `x`: Vector of unknown quantities (Scout, Kamikaze, Cargo)
- `b`: Vector of total available materials

**Solution Method:**
1. Check if system has unique solution using determinant: `det(A) ≠ 0`
2. Solve using `np.linalg.solve(A, b)`
3. Verify solution by checking: `A @ x == b`

**Key Concepts:**
- Matrix determinant
- System of linear equations
- Matrix inversion (via `np.linalg.solve`)
- Solution verification

---

### Task 4: Trend Monitoring (`task4_trand_monitoring.ipynb`)

**Objective:** Predict server CPU loading using linear regression (least squares).

**Problem:** Given historical CPU usage data over time, predict future server loading.

**Linear Model:**
```
y(t) = kt + b
```

Where:
- `y(t)`: CPU usage at time `t`
- `k`: Slope (rate of change)
- `b`: Y-intercept (baseline usage)

**Solution Method:**
1. Build design matrix: `A = [t, 1]` using `np.column_stack([t, np.ones(len(t))])`
2. Solve using least squares: `np.linalg.lstsq(A, y)`
3. Extract coefficients `k` and `b`
4. Predict: `y_predicted = k * t_new + b`
5. Clamp predictions to valid range [0, 100]% for CPU usage

**Key Concepts:**
- Least squares regression
- Design matrix construction
- Linear prediction
- Value clamping for bounded outputs

**Note:** Linear regression can predict values outside [0, 100]% range, so predictions are clamped to valid CPU usage bounds.

---

## Common NumPy Operations Used

- `np.array()`: Create arrays
- `np.dot()`: Dot product / matrix multiplication
- `np.linalg.norm()`: Vector/matrix norm
- `np.linalg.det()`: Matrix determinant
- `np.linalg.solve()`: Solve linear system
- `np.linalg.lstsq()`: Least squares solution
- `np.column_stack()`: Stack arrays as columns
- `np.clip()`: Clamp values to range
- `.T`: Matrix transpose
- `keepdims=True`: Preserve dimensions for broadcasting

## Running the Notebooks

You can run these notebooks in two ways:

### Option 1: Google Colab (Recommended)

1. **Upload to Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click `File` → `Upload notebook`
   - Select any `.ipynb` file from this repository

2. **Or open directly from GitHub:**
   - Upload the repository to GitHub
   - Open the notebook file on GitHub
   - Click "Open in Colab" button (if available)
   - Or use: `https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/task1_image_modification.ipynb`

3. **Run cells:**
   - Click on a cell and press `Shift + Enter` to run
   - Or use `Runtime` → `Run all` to execute all cells

**Advantages:**
- NumPy is pre-installed
- No local setup required
- Free GPU/TPU access (if needed)
- Easy sharing and collaboration

### Option 2: VS Code with Colab Extension

1. **Install VS Code:**
   - Download from [code.visualstudio.com](https://code.visualstudio.com/)

2. **Install Colab Extension:**
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X / Cmd+Shift+X)
   - Search for "Google Colab" extension
   - Install the official Google Colab extension

3. **Open Notebook:**
   - Open any `.ipynb` file in VS Code
   - Click "Open in Colab" button in the top-right corner
   - Or right-click the file → "Open in Colab"

4. **Run cells:**
   - Use the same keyboard shortcuts as Colab
   - `Shift + Enter` to run a cell

**Advantages:**
- Edit locally in VS Code
- Run in Colab cloud environment
- Best of both worlds

### For Interactive Tasks

Tasks 1 and 4 include interactive prompts. When running these:
- Enter values when prompted (or press Enter for default values)
- The notebook will process your input and display results

## Notes

- All tasks use NumPy for efficient matrix operations
- Tasks demonstrate both vector-based and matrix-based approaches where applicable
- Error handling and value validation are included where necessary
- Results are formatted for readability (e.g., 2 decimal places for predictions)
