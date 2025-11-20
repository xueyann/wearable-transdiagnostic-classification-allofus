import os
import pandas as pd
from google.cloud import bigquery
import time


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


    client = bigquery.Client()
    job = client.query(query)
    result = job.result()  # wait for the job to finish

    stop = time.time()

    print(f"Total time taken = {(stop-start/60)} minutes")