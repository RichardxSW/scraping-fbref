import pandas as pd
import holidays

tgl = pd.to_datetime('2025-07-10')

libur_id = holidays.Indonesia(years=tgl.year)

weekends = pd.date_range(start=f'{tgl.year}-01-01', end=f'{tgl.year}-12-31', freq='W-SAT').append(
    pd.date_range(start=f'{tgl.year}-01-01', end=f'{tgl.year}-12-31', freq='W-SUN')
)

# Menggabungkan libur nasional dengan Sabtu dan Minggu
all_holidays = set(libur_id.keys()).union(weekends.date)

is_holiday = tgl.date() in all_holidays

print(is_holiday)