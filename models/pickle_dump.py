import joblib
import sys

from train_classifier import main

if __name__ == "__main__":
    database_filepath, model_filepath = sys.argv[1:]
    model = main()
    joblib.dump(model, model_filepath)