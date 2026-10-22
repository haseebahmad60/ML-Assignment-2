# ML Assignment 2

End-to-end machine-learning workflows for Titanic passenger survival and Heart Disease prediction.

## Included

- `notebooks/Titanic_Passenger_Survival.ipynb` — data inspection, preprocessing, label encoding, SVC training, evaluation, and a sample prediction.
- `notebooks/Heart_Disease_Prediction.ipynb` — the same stages using exactly four inputs: `age`, `sex`, `cp`, and `thalach`. The UCI target `num=0` is mapped to **No disease** and `num=1–4` to **Disease present**.
- `data/` — the 100-row Titanic sample and the official 303-row UCI Cleveland dataset, with source notes.
- `titanic_model.pkl`, `heart_disease_model.pkl` — trained prediction bundles with encoders.
- `app.py`, `model_utils.py`, and `requirements.txt` — combined Streamlit app and dependencies.
- `tests/` — saved-model prediction and input validation checks.

## Run

```powershell
py -m pip install -r requirements.txt
py -m notebook
py -m streamlit run app.py
```

The app has separate Titanic and Heart Disease forms. The Heart Disease model is an educational project, not a medical diagnostic tool.

## Deployment

The public GitHub repository is `haseebahmad60/ML-Assignment-2`. In Streamlit Community Cloud, select branch `main` and `app.py` as the app file. Live app: [https://sp24-bse-149-ml-assignment-2.streamlit.app/](https://sp24-bse-149-ml-assignment-2.streamlit.app/). The verified link is included at the top of both notebooks.
