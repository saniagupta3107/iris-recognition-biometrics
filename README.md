# Iris Recognition Biometric System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, end-to-end iris recognition system built with Python. This project captures the unique patterns of the human iris to perform biometric authentication. It includes all the fundamental stages of a biometric pipeline: image preprocessing, iris segmentation, normalization, feature extraction, and matching.



---

## 📋 Key Features

* **Iris Segmentation:** Locates the iris and pupil boundaries in an eye image using the Hough Circle Transform.
* **Iris Normalization:** Implements Daugman's Rubber Sheet model to unwrap the circular iris region into a fixed-size rectangular block, making it invariant to pupil dilation.
* **Feature Extraction:** Utilizes a bank of Gabor filters to extract a unique binary template (iris code) from the normalized iris pattern.
* **Noise Masking:** Generates a mask to identify and exclude noisy regions (eyelids, eyelashes, reflections) from the matching process, improving accuracy.
* **Matching Algorithm:** Calculates the Hamming Distance between two iris codes to determine if they belong to the same person.
* **Enrollment & Verification:** A complete demonstration script (`demo.py`) that enrolls new irises and verifies them against the database.

---
## 🛠️ Technology Stack

* **Python 3**
* **OpenCV** (for all core image processing tasks)
* **NumPy** (for numerical operations and array manipulation)
* **Scikit-image** (for applying Gabor filters)

---

## 🚀 Setup and Installation

To run this project locally, follow these steps.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-github-username]/iris-recognition-biometrics.git
    cd iris-recognition-biometrics
    ```

2.  **Create and activate a virtual environment:**
    *On macOS/Linux:*
    ```bash
    python3 -m venv irisenv
    source irisenv/bin/activate
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file is included for easy installation.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt` file yet, you can create one by running `pip freeze > requirements.txt` in your activated environment after installing the packages.)*

4.  **Download the Dataset:**
    Download an iris dataset (like the [UPOL Iris Database](http://phoenix.inf.upol.cz/iris/)) and place all the `.png` image files directly inside the `data/` folder.

---

## ▶️ How to Run

The main script to demonstrate the system's functionality is `demo.py`. It will enroll an iris from a sample image and then test it against two other images (one from the same person and one from a different person).

From within the `src/` directory, run the following command:
```bash
python3 demo.py