import os.path

import streamlit
from pandas.errors import DatabaseError

from gesture_classification.monitoring.db import SqliteDatabase
from gesture_classification.monitoring.streamlit import training_tab
from gesture_classification.monitoring.streamlit.class_distribution_tab import (
    build_class_distribution_section,
)
from gesture_classification.monitoring.streamlit.data_distribution_tab import (
    build_data_distribution_sections,
)

MODULE_PATH = os.path.abspath(__file__)


def training_result(db: SqliteDatabase):
    try:
        training_tab.build_accuracy_section(db)
        training_tab.build_f1_scores_section(db)
        training_tab.build_confusion_matrix_section(db)

    except DatabaseError:
        streamlit.text(
            "Data bases not ready for monitoring. Please run training"
        )
        return


def data_distribution(db: SqliteDatabase):
    try:
        build_data_distribution_sections(db)

    except DatabaseError:
        streamlit.text(
            "Data bases not ready for monitoring. Please run training & predictions."
        )
        return


def class_distribution(db: SqliteDatabase):
    try:
        build_class_distribution_section(db)

    except DatabaseError:
        streamlit.text(
            "Data bases not ready for monitoring. Please run training & predictions."
        )
        return


def app(db: SqliteDatabase = None):
    streamlit.set_page_config(layout="wide")

    if db is None:
        db = SqliteDatabase()

    tab1, tab2, tab3 = streamlit.tabs(
        ["Training Results", "Data Distribution", "Class Distribution"]
    )
    with tab1:
        training_result(db)
    with tab2:
        data_distribution(db)
    with tab3:
        class_distribution(db)


if __name__ == "__main__":
    app()
