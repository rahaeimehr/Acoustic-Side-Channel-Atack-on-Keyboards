# Project Name

This repository is a Python project for processing data and training a machine learning model. Currently, it includes two key scripts:

- `data_preprocessing.py` for data preparation and pipeline creation.
- `train.py` for training the model using preprocessed data.

The project is in the initial stages, and this README serves as a template for further documentation.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
    - [Data Preprocessing](#data-preprocessing)
    - [Training](#training)
5. [Folder Structure](#folder-structure)
6. [Future Work](#future-work)

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

## DAta folder Structure

sample :

  
    SCAAI/
    ├── data/
    │   ├── sample1/
    │   │   └── words/
    │   ├── sample2/
    │   │   └── words/



## Usage

### Data Preprocessing
The `data_preprocessing.py` script processes the raw data and organizes it into the required format for training. Use the following commands:

```bash
python data_preproccesing.py --data_folders ../data/sample1/words ../data/sample2/words ../data/sample23/words --output_path ../data/pipeline 

```

### Training
The train.py script trains the model using preprocessed data. Use the following commands:

```bash
python train.py --data_path ../data/pipeline/tensors/
```
