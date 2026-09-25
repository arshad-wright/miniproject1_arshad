import pandas as pd

# The timetable repeats every week, so keep one row for each Day + Time_Slot
_table = pd.read_csv('scaled_occupancy_dataset.csv').drop_duplicates(['Day', 'Time_Slot'])
_expected = _table.set_index(['Day', 'Time_Slot'])['Expected_Zone_Level']

# The 4 class time slots, written in minutes since midnight (1:10 PM = 13:10)
SLOTS = {
    '8:30-10:15':  (8 * 60 + 30, 10 * 60 + 15),
    '10:30-12:15': (10 * 60 + 30, 12 * 60 + 15),
    '1:10-3:00':   (13 * 60 + 10, 15 * 60 + 0),
    '3:30-4:45':   (15 * 60 + 30, 16 * 60 + 45),
}


def find_slot(time_text):
    """Turn a time like '09:15' (24-hour) into a slot name, or None."""
    hours, minutes = time_text.split(':')
    now = int(hours) * 60 + int(minutes)
    for slot_name, (start, end) in SLOTS.items():
        if start <= now <= end:
            return slot_name
    return None


def get_expected_level(day, time_text):
    """Return (Expected_Zone_Level 0-4, a short note) for this day and time."""
    slot = find_slot(time_text)
    if slot is None:
        return 0, 'no class slot'
    if (day, slot) not in _expected.index:
        return 0, 'no timetable entry'
    return int(_expected[(day, slot)]), f'slot {slot}'
