# End-to-End Deep Learning Pipeline for Galaxy Morphological Classification

[Badge: Python 3.8+] [Badge: PyTorch] [Badge: Stable Diffusion & LoRA] [Badge: Data Science]

## Executive Summary
This project demonstrates a complete, end-to-end machine learning workflow, from handling raw crowdsourced datasets to deploying generative AI for data augmentation, and finally performing rigorous statistical evaluation. 

The pipeline classifies galaxy morphologies using image data from the Sloan Digital Sky Survey (SDSS) and Galaxy Zoo 2. A custom Convolutional Neural Network (CNN) was developed to automate classification, and various data augmentation techniques were systematically evaluated for their statistical significance. Notably, the pipeline integrates a generative AI approach, utilizing Stable Diffusion with Low-Rank Adaptation (LoRA) to explore the potential and current limitations of generative augmentation in scientific datasets.

A comprehensive report detailing the methodology, physics context, and analytical findings is available in the "galaxy-morphology-augmentation.pdf" file.

## Core Highlights
* Generative AI Augmentation: Implemented a custom synthetic data generation pipeline using Stable Diffusion v1.5 fine-tuned with LoRA weights.
* Automated Data Pipeline: Scripts to ingest, cross-reference, and preprocess raw SDSS images with complex metadata.
* Statistical Rigor: Developed post-training evaluation scripts to test the statistical significance of various augmentation methods using the Mann-Whitney U test.

## Key Results
* Final optimized model (combining Rotation, S+S, and Flipping) achieved an overall accuracy of 83.28%.
* Sub-class performance yielded 88.07% accuracy for non-disk galaxies and 79.19% for disk galaxies.
* Traditional spatial augmentations showed statistically significant improvements, while texture-based augmentations (e.g., Blur, Jitter) did not.
* Generative Augmentation Failure: The inclusion of AI-generated images resulted in a performance drop to 50.59%, equivalent to random guessing.

## Detailed Analysis: Why did AI-Gen Augmentation fail?
The poor performance of the generative augmentation approach was analyzed in detail to provide insights for future improvements:
1. Limited Morphological Diversity: While individual synthetic images appeared visually plausible, they exhibited a lack of diversity compared to the original SDSS dataset.
2. Mode Collapse/Averaging: Generative diffusion models tend to produce outputs clustered around an "average" representation, failing to capture the full structural spectrum of the training data, especially when using LoRA fine-tuning.
3. Prompt Design Constraints: The use of simple, single-word captions ("disk galaxy") limited the model's ability to learn complex morphological features.
4. Future Path: The analysis suggests that more advanced adaptation techniques like FouRA (Fourier Low Rank Adaptation) and improved descriptive prompt design could mitigate these diversity issues.

<img width="600" height="360" alt="accuracies_compare_last" src="https://github.com/user-attachments/assets/0cf83f95-ed09-4d10-bc37-8a33a2f95b2b" />

<img width="600" height="360" alt="ai_galaxy_examples" src="https://github.com/user-attachments/assets/25093058-5709-4667-ace6-97399cb77fe5" />

---
**Documentation:** A comprehensive report detailing the methodology, results, and analytical findings can be found in the [Final Report](./galaxy-morphology-augmentation.pdf).

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
