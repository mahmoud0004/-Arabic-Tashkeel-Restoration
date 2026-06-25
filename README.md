# Arabic Tashkeel Restoration — Notebook Project

This project is notebook-based and includes no delivered `.py` source files.

## Dataset path

Default path used inside the training notebook:

```text
D:\NLP Project\data\dataset
```

Place the Kaggle dataset `.txt` files inside that folder, or edit `DATA_DIR` in notebook 01.

## Files

- `01_train_tashkeel_model.ipynb` — train the PyTorch BiLSTM model.
- `02_evaluate_and_predict.ipynb` — load the trained model and test predictions.
- `03_streamlit_app_launcher.ipynb` — generate and launch the Streamlit UI from a notebook.
- `requirements.txt` — packages to install.

## Run order

```bash
pip install -r requirements.txt
```

Then run:

1. `01_train_tashkeel_model.ipynb`
2. `02_evaluate_and_predict.ipynb`
3. `03_streamlit_app_launcher.ipynb`

## Note

Streamlit normally runs from a Python script. This package does not include `.py` files, but notebook 03 creates a temporary `streamlit_app_generated.py` when you run it.
