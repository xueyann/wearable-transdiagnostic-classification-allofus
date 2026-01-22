import sys
sys.path.append('./')
from lib.get_data import fetch_query, replace_person_ids_in_query, fetch_df, persist_df
from lib.helpers import read_txt, chunked, get_files_in_dir, DefaultDict
import yaml
import pandas as pd
import os


def main():
    # fetch the configuration
    with open('conf/get_data.yaml') as f:
         config = yaml.safe_load(f)

    base_dir_sql = config['base_dir_sql']
    base_dir_bucket = config['base_dir_bucket']
    # Get person df from bucket
    person_df = fetch_df(config['person_df']['save_name'], subfolder = base_dir_bucket)

    # Grab the person_ids from person df and use it to get other data dataframes
    for _, fetch_dict in config['data_dfs'].items():
        # Only if we want to fetch the dataframes
        if fetch_dict['fetch']:
            sql = read_txt(f"{base_dir_sql}/{fetch_dict['sql']}") # read the sql query
            # fetch data for all patients if dataset is small enough
            if not fetch_dict['chunked']: 
                sql = replace_person_ids_in_query(sql, person_df['person_id'])
                df = fetch_query(sql)
                persist_df(df, fetch_dict['save_name'], subfolder = base_dir_bucket)
            # if there is an argument for chunked, fetch the data in chunks
            else:
                chunk_size = fetch_dict['chunked'] # get the chunk size
                i = 0
                # also keep a dictionary of which id corresponds to which file
                person_file_map = pd.DataFrame()
                for ids in chunked(person_df['person_id'].astype(str), chunk_size):
                    # append to person file map
                    person_file_map = pd.concat([person_file_map, 
                                                pd.DataFrame.from_dict({key: [i] for key in ids}, orient='index')])
                    sql_chunk = replace_person_ids_in_query(sql, ids)
                    df = fetch_query(sql_chunk)
                    # save dataframe with chunk number
                    persist_df(df, fetch_dict['save_name'].replace(".", f"_{i}."), subfolder = base_dir_bucket)
                    i+=1
                persist_df(person_file_map, f"person_file_map_{fetch_dict['save_name']}", subfolder = base_dir_bucket)
    
    # Grab the person_ids from person df and use it to get other data dataframes
    for _, fetch_dict in config['filtered_dfs'].items():
        # Only if we want to fetch the dataframes
        if fetch_dict['fetch']: 
            sql = read_txt(f"{base_dir_sql}/{fetch_dict['sql']}") # read the sql query
            # Get the filter string
            filter_file_list = get_files_in_dir(fetch_dict['filter_file_dir'],
                                                fetch_dict['filter_file_pattern'])
            # Create directory if not exist
            savedir = f"{base_dir_bucket}/{os.path.dirname(fetch_dict['save_name'])}"
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            for filter_file in filter_file_list:
                # Get file chunk id
                chunk_id = filter_file.split('_')[-1].split('.')[0]
                # if not int(chunk_id) in [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 
                #                          22, 23, 24, 25, 26, 27, 28, 29]:
                print(f"{chunk_id = }")
                # Get string of sleep dates
                print(f"Reading {filter_file} for filtering...")
                sleep_dates = pd.read_csv(filter_file)[['person_id', 'sleep_date']].drop_duplicates()
                sleep_dates = [tuple(x) for x in sleep_dates.values.tolist()]
                # fetch in 3 chunks bc otherwise sql query is too long
                n_parts = 5
                df = []
                start = 0
                stop = int(len(sleep_dates)/n_parts)
                for _ in range(n_parts):
                    # # Build the UNION ALL part
                    # sleep_date_str = ','.join(f"({pid}, DATE '{d}')" for pid, d in sleep_dates[start:5])
                    structs = []
                    for i, (pid, dt) in enumerate(sleep_dates[start:stop]):
                        if i == 0:
                            structs.append(f"STRUCT({pid} AS person_id, DATE '{dt}' AS d)")
                        else:
                            structs.append(f"STRUCT({pid}, DATE '{dt}')")
                    unnest_sql = "SELECT * FROM UNNEST([\n    " + ",\n    ".join(structs) + "\n  ])"
                    # Format sql string
                    sql = sql.format_map(DefaultDict({'filter_string': unnest_sql}))
                    print(len(sql))
                    df.append(fetch_query(sql))
                    # Update indices
                    start = stop
                    stop = min(stop + int(len(sleep_dates)/n_parts), len(sleep_dates)-1)
                    # Concatenate the chunks together
                    df = pd.concat(df)
                    savename = fetch_dict['save_name'].format(i=chunk_id)
                    print(f"Saving to {savename}...\n-*50")
                    as_parquet = savename.endswith('.parquet')
                    persist_df(df, savename, subfolder = base_dir_bucket, as_parquet = as_parquet)

if __name__ == '__main__':
    main()