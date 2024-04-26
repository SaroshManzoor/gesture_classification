import plotly.express as px
import streamlit

from gesture_classification import config


def build_class_distribution_section(db):
    reference_data = db.read_table(config.REFERENCE_DATA_TABLE).set_index(
        "index"
    )
    prediction_data = db.read_table(config.PREDICTION_TABLE).drop(
        columns=["index"]
    )
    streamlit.header("Class Distribution", divider="rainbow")

    streamlit.subheader("Training Data")

    ref_class_dist = reference_data["target"].value_counts()
    fig = px.bar(
        y=ref_class_dist.values,
        x=ref_class_dist.index,
    )
    fig.update_xaxes(range=[0.5, 8.5], tickmode="linear",
                     title="Target Class")

    fig.update_yaxes(title="Count")

    fig.layout.height = 400
    streamlit.plotly_chart(fig, use_container_width=True)

    ui_columns = streamlit.columns(len(prediction_data.columns))

    for index, column in enumerate(prediction_data.columns):
        with ui_columns[index]:
            class_dist = prediction_data[column].value_counts()
            streamlit.subheader(column)
            fig = px.bar(
                y=class_dist.values,
                x=class_dist.index,
            )

            fig.update_xaxes(range=[0.5, 8.5], tickmode="linear",
                             title="Predicted Class")

            fig.update_yaxes(title="Count")

            fig.layout.height = 400
            streamlit.plotly_chart(fig, use_container_width=True)
