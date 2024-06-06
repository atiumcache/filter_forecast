from argparse import ArgumentParser

import pandas as pd
import os
import subprocess
import datetime
import LSODA_forecast
import particle_filter


def process_args():
    """
    Processes command line arguments.

    :return Namespace: Contains the parsed command line arguments.
    """
    parser = ArgumentParser(
        description="Runs a particle filter over the given state's data."
    )
    parser.add_argument("state_code", help="state location code from 'locations.csv'")
    parser.add_argument(
        "forecast_start_date", help="day to forecast from. ISO 8601 format.", type=str
    )
    return parser.parse_args()


def get_population(state_code: str) -> int:
    """Return a state's population."""
    df = pd.read_csv("./datasets/state_populations.csv")
    try:
        population = df.loc[df["state_code"] == int(state_code), "population"].values
        return population[0]
    except:
        return None


def get_previous_80_rows(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a data frame containing 80 rows of a state's hospitalization data.
    Data runs from input date to 79 days prior.

    :param df: A single state's hospitalization data.
    :param target_date: Date object in ISO 8601 format.
    :return: The filtered df with 80 rows.
    """
    df["date"] = pd.to_datetime(df["date"])
    df_sorted = df.sort_values(by="date")
    date_index = df_sorted[df_sorted["date"] == target_date].index[0]
    start_index = max(date_index - 80, 0)
    result_df = df_sorted.iloc[start_index : date_index + 1]
    result_df = result_df.drop(columns=["state", "date"], axis=1)

    return result_df


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
