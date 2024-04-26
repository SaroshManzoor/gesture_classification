RANDOM_SEED = 0

""" Training """
HELD_OUT_USERS = ["U7 "]  # Data from these will be used for validation
MIN_TIME_STEPS = 20
MAX_TIME_STEPS = 250

CNN_N_EPOCHS = 8
LSTM_FCN_N_EPOCHS = 8


""" DB """
# Table names
REFERENCE_DATA_TABLE = "reference_data"
INFERENCE_DATA_TABLE = "current_sample"
PREDICTION_TABLE = "predictions"
CONFUSION_TABLE = "confusion_matrices"
F1_TABLE = "f1_scores"
ACCURACY_TABLE = "accuracies"
