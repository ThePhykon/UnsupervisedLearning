# UnsupervisedLearning

This repository contains my simple Python implementations and experiments related to unsupervised learning algorithms, with a focus on neuroinspired computing methods such as Self-Organizing Maps (SOM) and Oja's learning rule. These are not fully-fledged or production-ready implementations, but rather educational examples created to illustrate the core ideas. The project was created as part of the "Neuroinspired Computing" seminar I attended at university.

## Project Overview

The code in this repository demonstrates key concepts in unsupervised learning, including:
- **Self-Organizing Maps (SOM):** Visualization and clustering of high-dimensional data.
- **Oja's Learning Rule:** A biologically plausible algorithm for principal component analysis.

Visualizations and results are provided in the `pictures/` directory.

## Setup & Installation

To run the code, it is recommended to use a Python virtual environment. Follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd UnsupervisedLearning
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the scripts:**
   ```bash
   cd python
   python som.py
   python oja_learning.py
   ```

## Seminar Context

This repository was developed as part of the "Neuroinspired Computing" seminar, where the focus was on exploring and implementing algorithms inspired by neural computation and unsupervised learning principles.