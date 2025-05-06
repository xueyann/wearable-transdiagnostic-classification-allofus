from lib.get_data import fetch_query, persist_df
from lib.helpers import read_txt
import yaml


def main():
    # fetch the configuration
    with open('conf/get_data.yaml') as f:
         config = yaml.safe_load(f)

    base_dir_sql = config['base_dir_sql']
    base_dir_bucket = config['base_dir_bucket']
    # Get person df and persist to bucket
    person_sql = read_txt(f"{base_dir_sql}/{config['person_df']['sql']}")
    person_df = fetch_query(person_sql)
    persist_df(person_df, config['person_df']['save_name'], subfolder = base_dir_bucket)

    return
    
if __name__ == '__main__':
    main()