# Dental-detection

This repository contains code for dental closed-mesh segmentation and metric learning using
PyTorch and PyTorch Geometric (PyG). It loads 3D mesh/pointcloud data (STL/PLY), applies
custom transformations, and trains a DGCNN-based metric / segmentation model.

## Key files
- `src/train.py` - training loop and dataset wiring
- `src/data_loader.py` - dataset and DataLoader helpers (uses `pyvista` to load meshes)
- `src/model.py` - model definition (uses `torch_geometric.nn`)
- `src/transformations.py` - dataset transforms (uses `sklearn` PCA)
- `src/config.py` - configuration constants

## Dependencies
Primary dependencies are listed in `requirements.txt`. The core requirements are:

- torch (PyTorch)
- torchvision
- torch-geometric (PyG) and its binary dependencies
- numpy (<2.0)
- scipy
- scikit-learn
- pyvista
- tqdm

Important: PyTorch Geometric (PyG) is not always installable via a single pip line because it
depends on prebuilt wheels that match your PyTorch version and CUDA toolkit. Follow the
official PyG install guide for your platform: https://pytorch-geometric.readthedocs.io/

### Example (CPU-only) install
Below is an example sequence you can run on macOS / Linux if you only need CPU builds. Adapt
the `torch`/`torchvision` versions for your environment.

```bash
# Create & activate a venv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install core packages
pip install -U pip
pip install "torch>=1.13.1" "torchvision>=0.14.1" --extra-index-url https://download.pytorch.org/whl/cpu

# Install PyG - use CPU wheels matching torch here. Check PyG docs for the exact command.
pip install torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

# Install the rest
pip install -r requirements.txt
```

If you have CUDA, pick PyTorch wheels that match your CUDA version and then install
PyG wheels built for that same PyTorch+CUDA combination.

## Quick start
1. Place your dataset under the `data/` directory (STL/PLY supported by `pyvista`).
2. Edit `src/config.py` to set `DATA_DIR`, `DEVICE`, and training hyperparameters if needed.
3. Run training:

```bash
python src/train.py
```

## Notes & Troubleshooting
- If you hit import errors for `torch_geometric` or one of its subpackages, install the
	binary dependencies for PyG first (`torch-scatter`, `torch-sparse`, `torch-cluster`,
	`torch-spline-conv`) using the wheels from https://data.pyg.org/whl/.
- If `pyvista` fails to read meshes, ensure VTK is available for your environment; prefer
	installing `pyvista` via pip in a fresh venv.

## License
Add your license here.

```
# Dental-detection