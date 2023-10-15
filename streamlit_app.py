import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
import streamlit as st


@st.cache_resource
def return_trained_model():
    """Train a gradient boosting regression model using data with the following schema:

    - target: float
    - float_feature: float
    - discrete_feature: int
    - toggle_feature: string having "Yes"/"No" as value
    - categorical_feature: string with 4 unique categories
    """
    feature_names = [
        "float_feature",
        "discrete_feature",
        "toggle_feature",
        "categorical_feature",
    ]
    cat_features = ["toggle_feature", "categorical_feature"]

    data = pd.read_csv("sample_data.csv")
    X = data[feature_names]
    y = data["target"]

    model = make_pipeline(
        make_column_transformer(
            (OneHotEncoder(), cat_features), remainder="passthrough"
        ),
        GradientBoostingRegressor(),
    )
    model.fit(X, y)

    return data, model


def render_app():
    st.markdown(
        """
        #### This sample app serves an ML model with the ff features:
        * target: `float`
        * float_feature: `float` - value supplied using `st.number_input` widget
        * discrete_feature: `int` - value supplied using `st.slider` widget
        * toggle_feature: `string` (Yes/No) - value supplied using `st.radio` widget
        * categorical_feature: `string` - value supplied using `st.selectbox` widget
        """
    )
    data, model = return_trained_model()
    float_feature = st.number_input("float_feature")
    discrete_feature = st.select_slider("discrete_feature", range(4))
    toggle_feature = st.radio("Yes/No", ["Yes", "No"])
    categorical_feature = st.selectbox(
        "categorical_feature", data["categorical_feature"].sort_values().unique()
    )

    if st.button("Predict"):
        X = pd.DataFrame(
            {
                "float_feature": float_feature,
                "discrete_feature": discrete_feature,
                "toggle_feature": toggle_feature,
                "categorical_feature": categorical_feature,
            },
            index=[0],
        )
        st.session_state["prediction"] = round(model.predict(X)[0], 2)

    if st.session_state["prediction"] is not None:
        st.markdown(f"#### Target Prediction: {st.session_state['prediction']}")


if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

render_app()
