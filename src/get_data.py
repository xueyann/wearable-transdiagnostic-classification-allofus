from lib.get_data import fetch_query, replace_person_ids_in_query, fetch_df, persist_df
from lib.helpers import read_txt, chunked
import yaml


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
                for ids in chunked(person_df['person_id'].astype(str), chunk_size):
                    sql_chunk = replace_person_ids_in_query(sql, ids)
                    df = fetch_query(sql_chunk)
                    # save dataframe with chunk number
                    persist_df(df, fetch_dict['save_name'].replace(".", f"_{i}."), subfolder = base_dir_bucket)
                    i+=1

    
if __name__ == '__main__':
    main()