import pandas as pd
import pandas_gbq as pd_gbq
import os
import subprocess
from lib.helpers import DefaultDict

def fetch_query(query):
    # Replace {my_bucket}
    query = query.format_map(DefaultDict({'my_bucket': os.getenv('WORKSPACE_CDR')}))
    # first get the workspace bucket name
    df = pd_gbq.read_gbq(
        query,
        dialect = "standard",
        use_bqstorage_api = ("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
        progress_bar_type = None)
    
    return df

def replace_person_ids_in_query(query, person_ids):
    return query.format_map(DefaultDict({'person_ids': ','.join([f"{x}" for x in person_ids])}))

def persist_df(df, fname, bucket_dest = os.getenv('WORKSPACE_BUCKET'), subfolder='data'):
    # temporarily save df to same workspace as notebook
    df.to_csv(f"./{subfolder}/{fname}", index=False)
    # copy csv file to the bucket
    args = ["gsutil", "cp", f"./{subfolder}/{fname}", f"{bucket_dest}/{subfolder}/"]
    output = subprocess.run(args, capture_output=True)
    # print output from gsutil
    output.stderr
    return

def fetch_df(fname, bucket_dest = os.getenv('WORKSPACE_BUCKET'), subfolder='data'):
    # copy csv file from the bucket to the current working space
    os.system(f"gsutil cp '{bucket_dest}/{subfolder}/{fname}' './{subfolder}/{fname}'")
    print(f'[INFO] {fname} is successfully downloaded into your working space')
    # save dataframe in a csv file in the same workspace as the notebook
    df = pd.read_csv(f"./{subfolder}/{fname}")
    return df

