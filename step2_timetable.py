import joblib
import pandas as pd

# ---------- 1. Load the timetable and the trained model ----------
timetable = pd.read_csv('scaled_occupancy_dataset.csv')
model = joblib.load('zone_model.pkl')

# The timetable repeats every week, so keep one row for each Day + Time_Slot
timetable = timetable.drop_duplicates(['Day', 'Time_Slot'])
expected_lookup = timetable.set_index(['Day', 'Time_Slot'])['Expected_Zone_Level']

# ---------- 2. The 4 class time slots, written in minutes since midnight ----------
# Example: 8:30 = 8*60 + 30 = 510 minutes. "1:10" in your timetable means 1:10 PM = 13:10.
SLOTS = {
    '8:30-10:15':  (8 * 60 + 30, 10 * 60 + 15),
    '10:30-12:15': (10 * 60 + 30, 12 * 60 + 15),
    '1:10-3:00':   (13 * 60 + 10, 15 * 60 + 0),
    '3:30-4:45':   (15 * 60 + 30, 16 * 60 + 45),
}


def find_slot(time_text):
    """Turn a time like '09:15' (24-hour) into the matching slot name, or None."""
    hours, minutes = time_text.split(':')
    now = int(hours) * 60 + int(minutes)
    for slot_name, (start, end) in SLOTS.items():
        if start <= now <= end:
            return slot_name
    return None  # break time, lunch, or outside college hours


def get_expected_level(day, time_text):
    """Return the Expected_Zone_Level (0-4) for this day and time."""
    slot = find_slot(time_text)
    if slot is None:
        return 0, 'no class slot at this time'
    if (day, slot) not in expected_lookup.index:
        return 0, 'no timetable entry (weekend?)'
    return int(expected_lookup[(day, slot)]), f'slot {slot}'


def decide(expected, actual):
    """Ask the model what action to take."""
    data = pd.DataFrame([[expected, actual]],
                        columns=['Expected_Zone_Level', 'Actual_Zone_Level'])
    return model.predict(data)[0]


# ---------- 3. Try it out ----------
print("Type 'q' as the day to quit.\n")
while True:
    day = input("Day (e.g. Monday): ").strip().capitalize()
    if day == 'Q':
        break
    time_text = input("Time in 24-hour format (e.g. 09:00 or 14:00): ").strip()
    actual = int(input("Actual level (0-4, we type it by hand for now): "))

    expected, note = get_expected_level(day, time_text)
    action = decide(expected, actual)

    print(f"\n  Timetable says : Expected level = {expected}  ({note})")
    print(f"  You typed      : Actual level   = {actual}")
    print(f"  MODEL DECISION : {action}\n")
