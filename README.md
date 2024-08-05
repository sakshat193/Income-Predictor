# Income Predictor

This project is an income prediction application built using Streamlit and a Jupyter Notebook. The application predicts whether an individual's income is greater than $50,000 based on their demographic characteristics.

## Project Structure

- `app.py`: The main Streamlit application script.
- `income predictor.ipynb`: Jupyter Notebook containing the data analysis and model training.

## Installation

To run this project, you'll need Python installed on your machine along with the necessary libraries. Follow the steps below to set up the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Income-Predictor.git
    cd Income-Predictor
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

4. Place the required model files (`xgb_model.pkl`, `scaler.pkl`, `label_encoders.pkl`) in the project directory.

## Usage

1. **Running the Streamlit App:**

    To run the Streamlit app, use the following command:
    ```bash
    streamlit run app.py
    ```

    This will start the Streamlit server, and you can interact with the app through your web browser.

2. **Exploring the Jupyter Notebook:**

    Open the Jupyter Notebook using Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab income\ predictor.ipynb
    ```
    This notebook contains the data analysis and model training process.

## Features

- **Streamlit App:**
  - User-friendly interface for predicting income.
  - Various input fields for demographic characteristics.
  - Real-time prediction and result display.

- **Jupyter Notebook:**
  - Detailed data analysis.
  - Model training and evaluation using XGB classifiers.
  - Visualization of data insights.

## Data

The dataset used for this project includes various demographic characteristics such as age, workclass, education, marital status, occupation, and more. The target variable is whether the individual's income is greater than $50,000.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

This project was created using Streamlit, Pandas, scikit-learn, and XGBoost.
