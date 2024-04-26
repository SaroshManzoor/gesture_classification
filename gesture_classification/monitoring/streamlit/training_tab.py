import numpy as np
import streamlit
from plotly import express as px

from gesture_classification import config


def build_accuracy_section(db):
    streamlit.header("Validation Accuracies", divider="rainbow")

    accuracy_table = db.read_table(config.ACCURACY_TABLE).set_index("index")
    columns = streamlit.columns(len(accuracy_table), gap="large")

    for index, (classifier, accuracy) in enumerate(
        accuracy_table["Accuracy"].items()
    ):
        with columns[index]:
            streamlit.metric(classifier, value=round(accuracy, 2))

    # Gap between sections
    streamlit.markdown("#")
    streamlit.markdown("#")


def build_f1_scores_section(db):
    streamlit.header("F1 Scores", divider="rainbow")

    f1_table = db.read_table(config.F1_TABLE).set_index("index")
    f1_table.rename_axis("Classes", axis=1, inplace=True)
    f1_table.rename_axis("", axis=0, inplace=True)

    f1_table = np.round(f1_table, 2)
    heat_map = px.imshow(
        f1_table, text_auto=True, color_continuous_scale="greens"
    )

    heat_map.layout.width = 1200
    heat_map.layout.height = 300
    heat_map.update(layout_coloraxis_showscale=False)

    streamlit.plotly_chart(heat_map, use_container_width=True)

    # Gap between sections
    streamlit.markdown("#")
    streamlit.markdown("#")


def build_confusion_matrix_section(db):
    streamlit.header("Confusion Matrices", divider="rainbow")
    streamlit.markdown("#")

    confusion_tables = db.read_table(config.CONFUSION_TABLE).set_index(
        "Classifier"
    )

    classifiers = confusion_tables.index.get_level_values(0).unique()

    columns = streamlit.columns(len(classifiers), gap="medium")

    for index, classifier in enumerate(classifiers):
        with columns[index]:
            streamlit.subheader(classifier, divider=None)

            confusion_matrix = confusion_tables.loc[classifier].set_index(
                "level_1"
            )
            confusion_matrix.rename_axis("True Class", axis=0, inplace=True)
            confusion_matrix.rename_axis("Predicted Class", axis=1, inplace=True)

            heat_map_confusion = px.imshow(
                confusion_matrix, text_auto=True, color_continuous_scale="greens",
                title=" "
            )
            heat_map_confusion.update(layout_coloraxis_showscale=False)
            heat_map_confusion.update_layout(title_x=0,
                                             margin=dict(l=0, r=0, t=20, b=0),

                                             )
            heat_map_confusion.update_xaxes(tickmode="linear")

            streamlit.plotly_chart(heat_map_confusion, use_container_width=True)
