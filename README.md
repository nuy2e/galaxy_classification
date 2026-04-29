# Galaxy Classification

This project focuses on classifying galaxy morphologies utilizing image data from the Sloan Digital Sky Survey (SDSS) and crowdsourced labels from Galaxy Zoo 2. The complete pipeline encompasses data preparation, image processing, deep learning model training (with optional LoRA adaptation), and statistical result evaluation.

**Documentation:** A comprehensive report detailing the methodology, results, and analytical findings can be found in the [Final Report](./galaxy-morphology-augmentation.pdf).

---

## Project Structure

```text
.
├── Results/                         # Output directories for logs and accuracy metrics
│   ├── accuracy_results/
│   ├── example_model_log.csv
│   └── test_log.csv
│
├── analysis/                        # Statistical analysis and evaluation scripts
│   ├── Utest.py
│   ├── find_voterate.py
│   └── vote_rate.csv
│
├── data_preparation/                # Data loading, preprocessing, and curation
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
├── model_trained/                   # Directory for saving and loading model weights
│   └── example_model.pth            
│
├── src/                             # Core execution scripts
│   ├── galaxy_classification_eval.py   
│   └── galaxy_classification_train.py  
│
├── galaxy-morphology-augmentation.pdf 
└── README.md
```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/nuy2e/galaxy_classification.git](https://github.com/nuy2e/galaxy_classification.git)
cd galaxy_classification
```

### 2. Install Dependencies

Ensure you have Python 3.8 or higher installed. Install the required packages using pip:

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn tqdm
```
*(Note: If a `requirements.txt` file is generated later, you may alternatively use `pip install -r requirements.txt`)*

### 3. Data Acquisition

Download the external datasets referenced in the repository text files:
* `data_preparation/raw_images/images_download_link.txt`
* `data_preparation/initial_raw_data/link_to_galaxy_zoo_2.txt`

**Directory Placement:**
* Extract and place all image files directly into the `data_preparation/raw_images/` directory (do not use subdirectories).
* Place `gz2_filename_mapping.csv` and `gz2_hart16.csv.gz` into the `data_preparation/initial_raw_data/` directory.

### 4. Data Processing pipeline

Execute the data preparation scripts sequentially to format the dataset for training:
1. **Categorization:** Run `sort_galaxy.py`. This script cross-references the raw images with the Galaxy Zoo 2 metadata to categorize the data.
2. **Data Splitting:** Run `split_data.py`. This partitions the sorted dataset into distinct training, validation, and testing subsets.

### 5. Synthetic Data Generation (Optional)

To supplement the training data using AI-generated images:
* Execute `image_generator.py`, which utilizes Stable Diffusion to generate synthetic galaxy images.
* **LoRA Integration:** If utilizing Low-Rank Adaptation (LoRA) models, place the respective `.safetensors` files into `data_preparation/lora_model/` and ensure LoRA support is enabled within your training configuration.

---

## Usage

### Model Training

To initiate the training sequence, run the following command from the root directory:

```bash
python src/galaxy_classification_train.py
```

### Model Evaluation

To evaluate the trained model against the test dataset, execute:

```bash
python src/galaxy_classification_eval.py
```

---

## Analysis

Post-training analysis can be conducted utilizing the scripts located in the `analysis/` directory:
* `find_voterate.py`: Extracts and computes the vote rate from the Galaxy Zoo 2 metadata.
* `Utest.py`: Executes statistical significance testing (e.g., Mann-Whitney U test) on the results.

All generated logs and accuracy reports are automatically routed to the `Results/` directory.

---

## Configuration Notes

Before execution, please verify all file paths and hyperparameters within the scripts. Depending on your local environment and hardware constraints, these parameters may require manual adjustment.
