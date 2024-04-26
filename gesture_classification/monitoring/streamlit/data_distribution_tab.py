import pandas as pd
import streamlit
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from gesture_classification import config


def build_data_distribution_sections(db):
    reference_data = db.read_table(config.REFERENCE_DATA_TABLE).set_index(
        "index"
    )
    inference_data = db.read_table(config.INFERENCE_DATA_TABLE).set_index(
        "index"
    )
    build_average_distribution_section(inference_data, reference_data)
    build_std_distribution_section(inference_data, reference_data)


def build_std_distribution_section(inference_data, reference_data):
    streamlit.header(
        "Standard Deviation of acceleration per gesture", divider="rainbow"
    )
    streamlit.subheader("Training")
    _build_distribution_section(reference_data, column_substring="Std")
    streamlit.subheader("Inference")
    _build_distribution_section(inference_data, column_substring="Std")


def build_average_distribution_section(inference_data, reference_data):
    streamlit.header("Average acceleration per gesture", divider="rainbow")
    streamlit.subheader("Training")
    _build_distribution_section(reference_data, column_substring="Avg")
    streamlit.subheader("Inference")
    _build_distribution_section(inference_data, column_substring="Avg")


def _build_distribution_section(
    reference_data: pd.DataFrame, column_substring: str = "Avg"
) -> None:
    avg_columns = [
        col for col in reference_data.columns if column_substring in col
    ]
    fig = make_subplots(rows=1, cols=len(avg_columns), subplot_titles=avg_columns)

    for index, column in enumerate(avg_columns):
        fig.add_trace(
            go.Histogram(x=reference_data[column], name=column, nbinsx=50),
            row=1,
            col=index + 1,

        )

    fig.layout.width = 1500
    fig.layout.height = 450
    streamlit.write(fig)
