from model_utils import ensure_models, predict_bundle


def test_titanic_prediction_uses_saved_bundle():
    titanic, _ = ensure_models()
    assert titanic["feature_columns"] == ["PClass", "Gender", "Sibling", "Embarked"]
    result = predict_bundle(titanic, {
        "PClass": "Third", "Gender": "Female", "Sibling": "Zero", "Embarked": "Southampton"
    })
    assert result in {"Yes", "No"}


def test_heart_prediction_uses_exactly_four_saved_features():
    _, heart = ensure_models()
    assert heart["feature_columns"] == ["age", "sex", "cp", "thalach"]
    result = predict_bundle(heart, {"age": 54, "sex": 1, "cp": 4, "thalach": 150})
    assert result in {"Disease present", "No disease"}


def test_prediction_rejects_missing_or_extra_fields():
    titanic, _ = ensure_models()
    try:
        predict_bundle(titanic, {"PClass": "Third"})
    except ValueError as error:
        assert "exactly these input fields" in str(error)
    else:
        raise AssertionError("invalid input fields were accepted")
