import os
import pandas as pd
from google.cloud import bigquery
import time
import argparse


if __name__ == '__main__':

    start = time.time()

    dataset = os.environ['WORKSPACE_CDR']
    bucket = os.environ['WORKSPACE_BUCKET']


    query = f"""
        EXPORT DATA
            OPTIONS(
                uri='{bucket}/data/02_intermediate/fitbit_data/*.parquet',
                format='PARQUET',
                overwrite=true
            )
        AS
        SELECT 
            ac.person_id as person_id,
            ac.`datetime` as `datetime`,
            ac.steps as steps,
            hr.heart_rate_value as heart_rate,
            sl.sleep_date,
            sl.is_main_sleep,
            sl.level,
            sl.duration_in_min 
        FROM
            {dataset}.steps_intraday ac
        JOIN
            {dataset}.heart_rate_minute_level hr
        ON
            (
                ac.person_id = hr.person_id
            AND
                ac.`datetime` = hr.`datetime`
            )
        LEFT JOIN
            {dataset}.sleep_level sl  
        ON
            (   
                ac.person_id = sl.person_id
            AND
                ac.`datetime` = sl.start_datetime
            );
    """

    

    query_summary = f"""
        EXPORT DATA
            OPTIONS(
                uri='{bucket}/data/02_intermediate/fitbit_data_summary/*.parquet',
                format='PARQUET',
                overwrite=true
            )
        AS
        SELECT 
            hr.person_id,
            hr.date,
            hr.fat_burn_min,
            hr.cardio_min,
            hr.peak_min,
            hr.out_of_range_min,
            ac.activity_calories,
            ac.calories_bmr,
            ac.calories_out,
            ac.elevation,
            ac.fairly_active_minutes,
            ac.floors,
            ac.lightly_active_minutes,
            ac.marginal_calories,
            ac.sedentary_minutes,
            ac.steps,
            ac.very_active_minutes,
            sl.is_main_sleep,
            sl.minute_in_bed,
            sl.minute_asleep,
            sl.minute_after_wakeup,
            sl.minute_awake,
            sl.minute_restless,
            sl.minute_deep,
            sl.minute_light,
            sl.minute_rem,
            sl.minute_wake,
        FROM
            (SELECT
                person_id,
                date,
                IFNULL(SUM(CASE WHEN zone_name = 'Fat Burn'      THEN minute_in_zone END), 0) AS fat_burn_min,
                IFNULL(SUM(CASE WHEN zone_name = 'Cardio'        THEN minute_in_zone END), 0) AS cardio_min,
                IFNULL(SUM(CASE WHEN zone_name = 'Peak'          THEN minute_in_zone END), 0) AS peak_min,
                IFNULL(SUM(CASE WHEN zone_name = 'Out of Range'  THEN minute_in_zone END), 0) AS out_of_range_min
            FROM
                {dataset}.heart_rate_summary
            GROUP BY 
                person_id, date) hr
        JOIN
            {dataset}.activity_summary ac
        ON
            hr.person_id = ac.person_id
        AND
            hr.date = ac.date
        JOIN
            {dataset}.sleep_daily_summary sl
        ON
            hr.person_id = sl.person_id
        AND
            hr.date = sl.sleep_date"""
    
    print('starting')
    client = bigquery.Client()
    job = client.query(query_summary)
    result = job.result()  # wait for the job to finish

    stop = time.time()

    print(f"Total time taken = {(stop-start)/60} minutes")