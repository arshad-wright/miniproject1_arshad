import joblib
import pandas as pd

# 1. Load the trained model (zone_model.pkl must be in the same folder as this file)
model = joblib.load('zone_model.pkl')


def decide(expected, actual):
    """Give the model two numbers (0-4) and return its decision."""
    # The model was trained with column names, so we give it the same names
    data = pd.DataFrame([[expected, actual]],
                        columns=['Expected_Zone_Level', 'Actual_Zone_Level'])
    return model.predict(data)[0]


# 2. Try some situations and print what the model decides
tests = [
    (0, 0),  # nothing scheduled, nobody there
    (4, 0),  # big class expected, nobody there yet
    (4, 1),  # big class expected, very few people
    (2, 2),  # medium class, medium crowd
    (4, 3),  # big class, most students present
    (4, 4),  # big class, everyone present
]

print("Expected  Actual  ->  Decision")
for expected, actual in tests:
    print(f"   {expected}        {actual}     ->  {decide(expected, actual)}")
