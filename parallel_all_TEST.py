import os
import pandas as pd
import logging
import datetime
import ray
from multiprocessing import Pool
from filter_forecast.helpers import process_date

@ray.remote
def run_script_on_one_state(location_code, predict_from_dates, location_to_state, working_dir, logger):
    with Pool() as pool:
        tasks = [
            (location_code, date, location_to_state, working_dir, logger)
            for date in predict_from_dates["date"]
        ]
        pool.starmap(process_date, tasks)

def main():
    total_start_time = datetime.datetime.now()
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename="output.log", level=logging.INFO)

    # Initialize location mappings and 'predict-from' dates.
    locations = pd.read_csv("./datasets/locations.csv").iloc[1:]  # skip first row (national ID)
    location_to_state = dict(zip(locations["location"], locations["abbreviation"]))
    predict_from_dates = pd.read_csv("./datasets/predict_from_dates.csv")

    working_dir = os.getcwd()

    tasks = []
    for location_code in locations["location"].unique():
        tasks.append(run_script_on_one_state.remote(location_code, predict_from_dates, location_to_state, working_dir, logger))

    ray.get(tasks)

    total_end_time = datetime.datetime.now()
    elapsed_time = total_end_time - total_start_time
    elapsed_time_minutes = elapsed_time.total_seconds() / 60
    logger.info(f'All locations complete.\nTotal runtime: {round(elapsed_time_minutes, 2)} minutes.')

if __name__ == "__main__":
    main()
