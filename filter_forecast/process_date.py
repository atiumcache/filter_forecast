import os
import subprocess
import datetime
import LSODA_forecast
import particle_filter


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
