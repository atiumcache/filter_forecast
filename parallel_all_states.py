import os
import subprocess
from multiprocessing import Pool
import pandas as pd
import LSODA_forecast
import particle_filter
import logging
import datetime
import ray


def process_date(location_code, date, location_to_state, working_dir, logger):
    # Generate beta estimates from observed hospitalizations
    particle_filter.main(location_code, date)
    datetime_now = datetime.datetime.now()
    logger.info(
        f"Completed PF for location {location_code}: {date}. @ {datetime_now}"
    )

    # R script expects args: [working_dir, output_dir, location_code]
    # Generate beta forecasts
    output_dir = os.path.join(
        working_dir, f"datasets/beta_forecast_output/{location_code}/{date}"
    )
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run(
        ["Rscript", "./r_scripts/beta_trend_forecast.R", working_dir, output_dir, location_code],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"R script failed for location {location_code}, date {date}: {result.stderr}")
        return

    datetime_now = datetime.datetime.now()
    logger.info(
        f"Completed R script for location {location_code}: {date}. @ {datetime_now}"
    )

    # Generate hospitalization forecasts
    LSODA_forecast.main(location_to_state[location_code], location_code, date)
    logger.info(
        f"Completed LSODA_forecast for location {location_code}: {date}. @ {datetime_now}"
    )


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
