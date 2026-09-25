import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------
# 1. LOAD THE HISTORICAL DATA
# --------------------------------------------------

data = pd.read_csv("single_classroom_historical_occupancy_20pct.csv")

print("DATASET:")
print(data.head())

print("\nNumber of rows:", len(data))


# --------------------------------------------------
# 2. SELECT INPUTS AND OUTPUT
# --------------------------------------------------

X = data[
    [
        "Day",
        "Time_Slot",
        "Subject",
        "Expected_Students",
        "Is_Lab",
        "Is_Free"
    ]
]

y = data["Actual_Students"]


# --------------------------------------------------
# 3. DEFINE TEXT AND NUMBER COLUMNS
# --------------------------------------------------

categorical_columns = [
    "Day",
    "Time_Slot",
    "Subject",
    "Is_Lab",
    "Is_Free"
]

numeric_columns = [
    "Expected_Students"
]


# --------------------------------------------------
# 4. CONVERT TEXT DATA INTO NUMBERS
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# 5. CREATE THE RANDOM FOREST MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# 6. COMBINE PREPROCESSING + ML MODEL
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 7. SPLIT DATA INTO TRAINING AND TESTING DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 8. TRAIN THE MODEL
# --------------------------------------------------

pipeline.fit(X_train, y_train)

print("\nModel training completed!")


# --------------------------------------------------
# 9. TEST THE MODEL
# --------------------------------------------------

predictions = pipeline.predict(X_test)


print("\nACTUAL vs PREDICTED")

for actual, predicted in zip(y_test.head(10), predictions[:10]):
    print(
        "Actual:",
        actual,
        " Predicted:",
        round(predicted, 2)
    )


# --------------------------------------------------
# 10. CALCULATE MODEL PERFORMANCE
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 3))


# --------------------------------------------------
# 11. MAKE A NEW OCCUPANCY PREDICTION
# --------------------------------------------------

new_class = pd.DataFrame({
    "Day": ["Monday"],
    "Time_Slot": ["8:30-10:15"],
    "Subject": ["A"],
    "Expected_Students": [70],
    "Is_Lab": ["No"],
    "Is_Free": ["No"]
})


prediction = pipeline.predict(new_class)

print("\nNEW CLASSROOM PREDICTION")

print("Day: Monday")
print("Time: 8:30-10:15")
print("Subject: A")
print("Expected Students: 70")

print(
    "Predicted Actual Occupancy:",
    round(prediction[0], 2),
    "students"
)