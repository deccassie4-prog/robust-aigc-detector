# robust-aigc-detector
Robust detection of AI-generated images under real-world transformations like blur, compression, color adjustment, cropping, or rescaling

## Project Overview

This project aims to build a robust image-level detector that distinguishes AI-generated images (AIGC) from authentic images under real-world image transformations.

The system is designed to remain reliable after common post-processing operations such as JPEG compression, Gaussian blur, resizing, Gaussian noise, color jitter, and cropping.

## Key Objectives

- Detect whether an image is AI-generated or authentic.
- Evaluate robustness under realistic image transformations.
- Analyse false positives and false negatives.
- Provide a reproducible inference pipeline.

## Setup & Installation

### Requirements

- Python 3.10+
- Git
- PyTorch
- torchvision
- scikit-learn
- pandas
- Pillow
- NumPy

### Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd robust-aigc-detector
pip install -r requirements.txt
pip install -r requirements.txt
.venv\Scripts\activate

## Reproduction

### 1. Prepare the Environment

Create and activate a Python virtual environment, then install the required dependencies as described in the [Setup & Installation](#setup--installation) section.

### 2. Prepare the Data

Download the permitted datasets and place the data in the appropriate local directories.

Training and development data should not include the validation data specified in the competition rules.

### 3. Run Inference

Run the inference script on a directory of images:

```bash
python scripts/predict.py --input_dir <IMAGE_DIRECTORY> --output <OUTPUT_JSON>
python evaluation/evaluate.py

## Limitations

Our prototype is designed for a hackathon-scale setting and has several limitations:

- **Limited generalisation:** Performance may vary for AI-generated images from generators or distributions that are not represented in the training data.
- **Transformation coverage:** We evaluate a selected subset of real-world transformations rather than every possible type of image manipulation.
- **False positives and false negatives:** No image-level detector is perfectly reliable. Authentic images may be classified as AIGC, while some AI-generated images may be classified as authentic.
- **Limited compute:** The model and experiments are constrained by hackathon-scale computational resources.
- **Evolving generative models:** Image generation techniques continue to improve, so detection performance may degrade as new generation methods emerge.

Given more time and resources, we would expand the diversity of training data, evaluate against a wider range of generators and transformations, and investigate stronger domain-generalisation and calibration methods.

## Team Contributions

This project was developed collaboratively by a five-member team.

### Zhang Yiran — Inference Pipeline & Model Execution

* Cloned the upstream MIRROR repository and verified the upstream MIT license.
* Set up the required environment and downloaded the required model weights.
* Organised the validation dataset into the required real and AI-generated image structure.
* Ran the original inference pipeline and established the clean-image baseline performance.
* Verified the total number of model parameters to demonstrate compliance with the competition requirement of fewer than 2 billion parameters.
* Developed `predict.py`, which takes an image directory as input and outputs a JSON file containing `image_path` and `pred` for each image.
* Documented the environment setup and implementation issues for the rest of the team.

### Liu Zhenning — Robustness Evaluation

* Developed image transformation scripts covering JPEG compression, Gaussian blur, resizing, Gaussian noise, colour jitter, and centre cropping.
* Generated clean and transformed versions of the validation images.
* Ran batch inference after the inference environment was established.
* Produced a comparison of performance on clean versus transformed images.
* Shared per-image prediction results for subsequent error analysis.

### Chen Yizhen — Repository & Project Management

* Set up and maintained the GitHub repository and project directory structure.
* Established the README structure, including project overview, installation, reproduction steps, limitations, and team contributions.
* Defined the team's Git collaboration workflow, including branches, commits, pull requests, and file organisation.
* Integrated the final deliverables, including the inference script, evaluation pipeline, robustness results, error analysis, and documentation materials.
* Preserved the upstream MIT license and included appropriate attribution to the MIRROR project.

### Zhou Shuyuan — Error Analysis & Explainability

* Studied the two-stage MIRROR approach, with particular attention to residual-based comparison as an interpretable signal.
* Selected representative false positives and false negatives from per-image prediction results.
* Produced residual heatmap visualisations to analyse regions contributing to model decisions.
* Prepared the error analysis report, including representative cases and discussions of trade-offs such as robustness versus false-positive rates.

### Wang Zihan — Devpost Documentation & Demo Video

* Prepared the written Devpost project description, including the tools, models, libraries, and datasets used.
* Documented that the project uses pretrained weights for zero-shot inference and does not use the competition validation data for training.
* Designed the end-to-end demonstration flow, including image inputs, prediction scores, robustness evaluation results, and explainability visualisations.
* Recorded and edited the demo video.
* Uploaded the demo video to YouTube with public visibility and added the link to the Devpost submission and repository README.

## Project Workflow

The main development dependencies are:

1. **A → B:** Robustness evaluation depends on the inference environment and prediction pipeline being operational.
2. **A → C:** Repository documentation and reproducibility instructions depend on the final inference setup.
3. **B → D:** Error analysis depends on per-image prediction results from the evaluation pipeline.
4. **B + D → E:** The final demo depends on the robustness comparison and explainability results.
5. **A, B, D, E → C:** Final deliverables are integrated into the GitHub repository and README.

## Repository & Project Management

The repository is maintained using Git and GitHub. Final project components are integrated into the repository before submission. The project also preserves the upstream MIT license and includes appropriate attribution to the MIRROR project.


## Repository Structure

```text
robust-aigc-detector/
├── analysis/       # Error analysis, visualisations, and reports
├── assets/         # Images and demo-related assets
├── configs/        # Configuration files
├── evaluation/     # Robustness evaluation scripts and results
├── scripts/        # Executable scripts and data-processing utilities
├── src/            # Core project source code
├── .gitignore      # Files excluded from version control
├── CONTRIBUTING.md # Git and collaboration guidelines
├── LICENSE         # MIT License
├── README.md       # Project documentation
└── requirements.txt # Python dependencies
### Repository & Project Management

- Repository structure and collaboration workflow were established using Git and GitHub.
- Project documentation and README maintenance were coordinated throughout development.
- Final project components, evaluation results, error analysis, and demo materials were integrated into the repository.
## Acknowledgements

This project builds upon the MIRROR (Manifold Ideal Reference ReconstructOR) project for generalizable AI-generated image detection.

Upstream repository:
https://github.com/handsome-rich/MIRROR

Paper:
Ruiqi Liu et al., "MIRROR: Manifold Ideal Reference ReconstructOR for Generalizable AI-Generated Image Detection", arXiv:2602.02222, 2026.

Our work focuses on evaluating and improving the robustness of AI-generated image detection under realistic image transformations, including compression, blur, resizing, noise, color adjustment, and cropping.

The MIRROR codebase and pretrained models are used as an upstream component of our prototype. Please refer to the upstream repository for its original implementation and licensing information.