import cv2 as cv
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ==================================================
# 1. LOAD HISTORICAL DATA
# ==================================================

data = pd.read_csv(
    "single_classroom_historical_occupancy_20pct.csv"
)


# ==================================================
# 2. INPUT AND OUTPUT
# ==================================================

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


# ==================================================
# 3. COLUMNS
# ==================================================

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


# ==================================================
# 4. PREPROCESSING
# ==================================================

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


# ==================================================
# 5. RANDOM FOREST
# ==================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ==================================================
# 6. COMPLETE ML PIPELINE
# ==================================================

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# ==================================================
# 7. TRAIN MODEL
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

pipeline.fit(X_train, y_train)

print("ML MODEL TRAINED")


# ==================================================
# 8. CURRENT CLASSROOM INFORMATION
# ==================================================

current_class = pd.DataFrame({
    "Day": ["Monday"],
    "Time_Slot": ["8:30-10:15"],
    "Subject": ["A"],
    "Expected_Students": [70],
    "Is_Lab": ["No"],
    "Is_Free": ["No"]
})


# ==================================================
# 9. ML PREDICTION
# ==================================================

predicted_occupancy = pipeline.predict(current_class)

predicted_occupancy = round(
    predicted_occupancy[0]
)

print(
    "Predicted occupancy:",
    predicted_occupancy,
    "students"
)


# ==================================================
# 10. OPEN CAMERA
# ==================================================

cap = cv.VideoCapture(0)

width = 640
height = 480


# ==================================================
# 11. REMEMBER PREVIOUS ZONES
# ==================================================

previous_zones = set()


# ==================================================
# 12. CAMERA LOOP
# ==================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera could not be opened")
        break


    # ----------------------------------------------
    # Convert BGR image to HSV
    # ----------------------------------------------

    hsv = cv.cvtColor(
        frame,
        cv.COLOR_BGR2HSV
    )


    # ----------------------------------------------
    # RED COLOR RANGE
    # ----------------------------------------------

    lower_red1 = np.array(
        [0, 120, 70]
    )

    upper_red1 = np.array(
        [10, 255, 255]
    )

    lower_red2 = np.array(
        [170, 120, 70]
    )

    upper_red2 = np.array(
        [180, 255, 255]
    )


    # ----------------------------------------------
    # CREATE MASK
    # ----------------------------------------------

    mask1 = cv.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    mask2 = cv.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    mask = mask1 + mask2


    # ----------------------------------------------
    # FIND OBJECTS
    # ----------------------------------------------

    contours, hierarchy = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )


    current_zones = set()


    # ==================================================
    # 13. CHECK EVERY DETECTED OBJECT
    # ==================================================

    for contour in contours:

        area = cv.contourArea(contour)


        if area > 500:

            x, y, w, h = cv.boundingRect(
                contour
            )


            # Object center

            cx = x + w // 2
            cy = y + h // 2


            # ------------------------------------------
            # DETERMINE ZONE
            # ------------------------------------------

            if cx < width // 2 and cy < height // 2:

                zone = 1

            elif cx >= width // 2 and cy < height // 2:

                zone = 2

            elif cx < width // 2 and cy >= height // 2:

                zone = 3

            else:

                zone = 4


            current_zones.add(zone)


            # ------------------------------------------
            # DRAW RECTANGLE
            # ------------------------------------------

            cv.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            # ------------------------------------------
            # SHOW ZONE
            # ------------------------------------------

            cv.putText(
                frame,
                f"Zone {zone}",
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # ==================================================
    # 14. DISPLAY ML INFORMATION
    # ==================================================

    cv.putText(
        frame,
        f"Subject: A",
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv.putText(
        frame,
        f"Expected: 70",
        (20, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv.putText(
        frame,
        f"ML Predicted: {predicted_occupancy}",
        (20, 90),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ==================================================
    # 15. SHOW OCCUPIED ZONES
    # ==================================================

    zone_text = str(
        sorted(current_zones)
    )

    cv.putText(
        frame,
        f"Occupied zones: {zone_text}",
        (20, 120),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ==================================================
    # 16. DRAW CLASSROOM ZONES
    # ==================================================

    cv.line(
        frame,
        (width // 2, 0),
        (width // 2, height),
        (255, 255, 255),
        2
    )

    cv.line(
        frame,
        (0, height // 2),
        (width, height // 2),
        (255, 255, 255),
        2
    )


    # ==================================================
    # 17. SHOW WINDOWS
    # ==================================================

    cv.imshow(
        "Classroom AI",
        frame
    )

    cv.imshow(
        "Red Mask",
        mask
    )


    # ==================================================
    # 18. QUIT
    # ==================================================

    key = cv.waitKey(1)

    if key == ord("q"):
        break


# ==================================================
# 19. CLOSE CAMERA
# ==================================================

cap.release()

cv.destroyAllWindows()