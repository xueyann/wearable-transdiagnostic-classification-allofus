import yaml
import glob
import pandas as pd
from lib.get_data import persist_df

def main():
    # fetch the configuration
    with open('conf/map_phenotype.yaml') as f:
         config = yaml.safe_load(f)

    # Get the mapping file
    phecode_mapping = pd.read_csv(config['phecode_map'])
    pheinfo = pd.read_csv(config['pheinfo'])
    mapping = phecode_mapping.merge(pheinfo, on = 'phecode')

    # Get the condition dfs
    c_transform_config = config['condition_df_transform']
    for fpath in glob.glob(config['condition_dfs']):
        condition_df = pd.read_csv(fpath)
        # rename columns and discard unnecessary columns
        condition_df = condition_df.rename(columns = c_transform_config['col_rename'])
        condition_df[c_transform_config['keep_cols']]
        # rename the columns for joining with phecode mapping
        condition_df = condition_df.merge(mapping, on = c_transform_config['join_cols'])
        # save file to new directory with same name
        fname = fpath.split('/')[-1]
        persist_df(condition_df, fname, subfolder = c_transform_config['savepath'])
    
    return


if __name__ == '__main__':
    main()


