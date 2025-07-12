# Computer Vision Object Detection Project

A computer vision project focused on object detection and edge detection using YOLO models.

## Project Overview

This repository contains:
- **Training Notebook**: Complete training pipeline for object detection models
- **Evaluation & Detection Notebook**: Model evaluation and inference pipeline
- **Dataset**: Training and validation data with annotations
- **Edge Detections**: Results from edge detection analysis
- **Original Files**: Source videos and pre-trained models
- **Output Files**: Generated results and trained models

## Ways to Access and Run

### Option 1: Google Colab (Much more Recommended)

### Note - See my Explanation video at 

#### Video Explanation
**Link**: [Project Explanation Video](https://drive.google.com/file/d/1JXVL7Nqgsq2LokKDi5Y5udTMYGTCr0lt/view?usp=drive_link)

Watch the complete project explanation and demonstration.

### Note - Only Run the Evaluation and detection notebook as Training notebook just shows how i finetuned and trained the model 

#### Training Notebook
**Link**: [Training Notebook on Google Colab](https://colab.research.google.com/drive/1V9dAILT5cruINV5y4Fvfi5QuA4kjGM86?authuser=2)

#### Evaluation & Detection Notebook
**Link**: [Evaluation & Detection Notebook on Google Colab](https://colab.research.google.com/drive/1V9dAILT5cruINV5y4Fvfi5QuA4kjGM86?usp=sharing)


**Steps**:
1. Click the link above
2. Sign in to your Google account if prompted
3. Go through each cell and run them sequentially
4. Follow the instructions in the notebook for data upload and model training


### Option 2: Google Drive Resources

#### Complete Project Files
**Link**: [Complete Project on Google Drive](https://drive.google.com/drive/folders/1EURCjXZbgJ17IPolXhBdToj-dH4wCq4n?usp=drive_link)

Contains all project files including:
- Training and Evaluation notebooks
- Dataset, Edge Detections, Original Files, and Output Files
- Video Explanation

#### Custom Trained Model
**Link**: [My Custom Trained Model](https://drive.google.com/drive/folders/1Go5hnTki0gDKrSwWpW5fGZKgp3wfMiEC?usp=drive_link)

Download the trained model file (`trained_best.pt`) for immediate use.

#### Output Videos with Re-identification
**Link**: [Output Videos with Re-identification](https://drive.google.com/drive/folders/1wpaMwGTSqUv8HWFhjAr8K0dX4q3i1pF3?usp=drive_link)

View the processed videos showing object detection and re-identification results.

### Option 3: Local Setup

#### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required packages: `torch`, `torchvision`, `opencv-python`, `ultralytics`, `numpy`, `matplotlib`

#### Installation
```bash
# Clone the repository
git clone https://github.com/Ankit-03G/stealth_project.git
cd stealth_project

# Install dependencies
pip install torch torchvision opencv-python ultralytics numpy matplotlib jupyter

# Launch Jupyter
jupyter notebook
```

#### Running the Notebooks
1. Open `Training.ipynb` in Jupyter
2. Run all cells sequentially
3. Open `Evaluation_Detection.ipynb` for evaluation
4. Follow the instructions in each notebook

## Project Structure

```
Stealth_1/
├── Training.ipynb                 # Training pipeline
├── Evaluation_Detection.ipynb     # Evaluation and inference
├── Dataset/                       # Training data
│   ├── images/                    # Training images
│   ├── labels/                    # Annotation files
│   ├── classes.txt               # Class definitions
│   └── dataset.yaml              # Dataset configuration
├── Edge_Detections/              # Edge detection results
├── Orignal_files/                # Source videos and models
└── Output_Files/                 # Generated outputs
```

## Features

- **Object Detection**: YOLO-based training and inference
- **Edge Detection**: Advanced edge detection algorithms
- **Video Processing**: Support for video input/output
- **Model Evaluation**: Comprehensive evaluation metrics
- **Real-time Detection**: Live video processing capabilities

## Usage

1. **Training**: Use the Training notebook to train your custom object detection model
2. **Evaluation**: Use the Evaluation notebook to test your trained model
3. **Inference**: Run detection on new images/videos using the trained model

## Notes

- Large model files (`.pt` files) are excluded from this repository due to GitHub's file size limits
- Use Google Colab for GPU-accelerated training
- Ensure you have sufficient storage for dataset and model files

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the [MIT License](LICENSE).
