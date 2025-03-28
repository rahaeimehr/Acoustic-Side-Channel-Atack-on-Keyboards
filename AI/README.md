# Project Name

This repository is a Python project for processing data and training a machine learning model. Currently, it includes three key scripts:

- `raw_data_preprocessing.py` for data preparation and pipeline creation.
- `train_correct.py` for training the model using preprocessed data.
- `inference.py` for inference and evaluation if the model using preprocessed data.

The project is in the production stages, and this README serves as a documentation.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
    - [Data Preprocessing](#data-preprocessing)
    - [Training](#training)
5. [Parameters](#Parameters)
6. [Output](#output)

---

## Getting Started

To use this project, clone the repository and ensure you have Python installed with the required dependencies.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/project-name.git 
   ```
2. Install the required dependencies:
    ```bash
   pip install -r requirements.txt 
   ```

## Data folder Structure

Before running the pipeline, please ensure that your data follows the required structure for preprocessing.

make sure each sample folder name is a digit nothing else

sample :

  
    SCAAI/
    ├── data/
    │   ├── (digit)/
    │   │   ├── text.txt
    │   │   ├── text.wav
    │   │   └── any other other file is ok
    │   ├── (digit)/
    │   │   ├── text.txt
    │   │   ├── text.wav
    │   │   └── any other other file is ok

Make sure you have created the output and model directories as shown below before running the pipeline. 

  
    SCAAI/
    ├── <output dir name>/
    │   ├── model_data/
    │   ├── output/
    │   │   └── models/
    │   └── tensors/

## Usage

### Data Preprocessing
The `raw_data_preproccesing.py` script processes the raw data and organizes it into the required format for training. Use the following commands:

```bash
python raw_data_preproccesing.py --path ../data --output_path ../<output dir name>
```

### Training
The train_correct.py script trains the model using preprocessed data. Use the following commands:

```bash
python train_correct.py --data_path ../<output dir name>/tensors  --output_path ../../<output dir name>/output/models --save
```

### Inference
The inference.py script runs inference using a trained model and input data. Use the following command:

```bash
python infernce.py --path ../data --audio_number 3 --model_path ../<output dir name>/output/models/model_checkpoint.pth --output_path ../<output dir name>
```

## Parameters


The following arguments can be used with `raw_data_preproccesing.py` to control how the data is processed:

| Parameter             | Description                                                                                   | Default      |
|-----------------------|-----------------------------------------------------------------------------------------------|--------------|
| `--n_fft`             | Number of samples used for FFT                                                                | `2048`       |
| `--hop_length`        | Number of samples between successive frames                                                   | `512`        |
| `--n_mels`            | Number of Mel bands                                                                           | `64`         |
| `--path`              | Path containing audio and timestamps                                                          | **Required** |
| `--output_path`       | Path to save preprocessed data                                                                | **Required** |
| `--resampling_type`   | Type of resampling for balancing data (`majority`, `minority`, or `None`)                     | `'None'`     |
| `--resampling_order`  | Whether to resample data `before` or `after` the train-test split                             | `'after'`    |
| `--save`              | Save the preprocessed data to the specified output path                                       | `True`      |
| `--verbose`           | Verbose level: `0` = silent, `1` = basic, `2` = detailed                                      | `0`          |


The following arguments can be used when running `train_correct.py`:

| Parameter        | Description                                      | Default     |
|------------------|--------------------------------------------------|-------------|
| `--data_path`     | Path to the input data                          | **Required** |
| `--output_path`   | Path to save the preprocessed/trained output    | **Required** |
| `--save`          | Save the acc and loss figs                      | `False`     |
| `--device`        | Device to use for training/testing (`cuda` or `cpu`) | `'cuda'`    |


The following arguments can be used when running `inference.py`:

| Parameter         | Description                                                                 | Default      |
|-------------------|-----------------------------------------------------------------------------|--------------|
| `--n_fft`          | Number of samples used for FFT                                              | `2048`       |
| `--hop_length`     | Number of samples between successive frames                                 | `512`        |
| `--n_mels`         | Number of Mel bands                                                         | `64`         |
| `--path`           | Path containing audio and timestamps                                        | **Required** |
| `--audio_number`   | Index of the audio file to use from the given path                          | `0`          |
| `--model_path`     | Path to the trained model checkpoint (`.pt` file)                           | **Required** |
| `--show_plot`      | Show a plot of the waveform and predictions                                 | `False`      |
| `--threshold`      | Threshold value for classification                                          | `0.9`        |
| `--output_path`    | Path to save the inference output                                           | **Required** |

> ⚠️ **Important:**  
> The parameters `n_fft`, `hop_length`, and `n_mels` **must be the same** across **all scripts** (`preprocess.py`, `train_correct.py`, and `inference.py`) to ensure consistency in feature extraction.


## Output 

After running the inference script, the predicted timestamps will be saved to:

<output_directory>/predicted_timesteps.txt

> 📁 This file contains the timesteps where the model detected the target events or classes.