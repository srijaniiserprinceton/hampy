import numpy as np
from tqdm import tqdm

from hampy import load_data

if __name__=='__main__':
    # used defined start and end times in YYYY-MM-DD/hh:mm:ss format
    tstart = '2020-01-29/18:00:00'
    tend   = '2020-01-31/18:20:00'

    # setting up the data loading process [processing will happen one day at a time]
    span_data = load_data.span(tstart, tend)

    for day_idx in tqdm(span_data.Ndays):
        # loading in data for the current day
        span_data.start_new_day(day_idx)

