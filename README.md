# Galaxy Classification

This project focuses on classifying galaxy morphologies using image data from the Sloan Digital Sky Survey (SDSS) and labels from Galaxy Zoo 2. The pipeline includes data preparation, image processing, model training (with optional LoRA adaptation), and result evaluation.

---

## 📁 Project Structure

```
.
├── Results/                         # Logs and accuracy results
│   ├── accuracy_results/
│   ├── example_model_log.csv
│   └── test_log.csv
│
├── analysis/                        # Analysis scripts
│   ├── Utest.py
│   ├── find_voterate.py
│   └── vote_rate.csv
│
├── data_preparation/               # Data loading and preprocessing
│   ├── dataset/
│   │   └── notes.txt
│   ├── initial_raw_data/
│   │   └── link_to_galaxy_zoo_2.txt
│   ├── lora_model/
│   │   ├── E.safetensors
│   │   ├── S.safetensors
│   │   └── place_your_lora_model.txt
│   ├── raw_images/
│   │   └── images_download_link.txt
│   └── sorted_data/
│       ├── image_generator.py
│       ├── sort_galaxy.py
│       └── split_data.py
│
├── model_trained/
│   └── example_model.pth           # Pretrained model weights
│
├── src/
│   ├── galaxy_classification_eval.py   # Evaluation script
│   └── galaxy_classification_train.py  # Training script
│
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/nuy2e/galaxy_classification.git
cd galaxy_classification
```

### 2. Download Image Data

Download the files referenced in:
- `data_preparation/raw_images/images_download_link.txt`
- `data_preparation/initial_raw_data/link_to_galaxy_zoo_2.txt`

Then:
- Place all images directly under `data_preparation/raw_images/` from `images_download_link.txt`, (no subfolders).
- Place `gz2_filename_mapping.csv`(from `images_download_link.txt`) and `gz2_hart16.csv.gz`(from `link_to_galaxy_zoo_2.txt`) under `data_preparation/initial_raw_data/`.

### 3. Sort and Split Images

Run the scripts in the following order:
- `sort_galaxy.py`: Sorts and categorises galaxies based on label metadata. It references `raw_images/` and uses Galaxy Zoo classification data.
- `split_data.py`: Splits the dataset into training, validation, and test subsets. It references the `sorted_data/` output from the previous step.

### 4. (Optional) Generate AI-based Images

- `image_generator.py`: Uses Stable Diffusion to generate synthetic images.
- To use LoRA models:
  - Place your `.safetensors` files in `data_preparation/lora_model/`.
  - Reference them in the training script, ensuring LoRA support is enabled.

---

## 🛠️ Usage

### Train the Model

```bash
python src/galaxy_classification_train_git.py
```

### Evaluate the Model

```bash
python src/galaxy_classification_eval_git.py
```

---

## 📊 Analysis

- `analysis/find_voterate.py`: Computes the vote rate from Galaxy Zoo 2 metadata.
- `analysis/Utest.py`: Performs statistical significance tests.
- Output logs and accuracy reports are saved under `Results/`.

---

## 📝 Notes

- All paths and parameters may need to be modified in the scripts to suit your local setup.

---

## 📌 Requirements

Install the required Python packages:

- Python ≥ 3.8
- `torch`, `torchvision`
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tqdm`

You can install them with:

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, you may need to create one by listing the libraries used in your environment.
