import pandas as pd
import numpy as np
import yaml


def process_demo(config):
    """
    Process demographics information
    Note that this function does not compute age"""
    fpath = config['fpath']
    excl_values = config['excl_values']
    save_path = config['save_path']

    person_df = pd.read_csv(fpath)

    # Replace certain answers with nan
    for k, v in excl_values.items():
        person_df.loc[person_df[k].isin(v), k] = np.nan

    person_df.to_csv(save_path, index=False)
    
    return person_df

def process_osa_device(config):
    fpath = config['fpath']
    save_path = config['save_path']

    device_df = pd.read_csv(fpath)

    # Get the earliest date of OSA device prescription
    osa_device = device_df[['person_id', 'device_exposure_start_datetime']]\
            .groupby('person_id', as_index=False ).min().rename(columns={'device_exposure_start_datetime': 
                                                                         'osa_device_date'})
    osa_device.to_csv(save_path, index=False)

    return osa_device


def process_physical_measurement(config):

    # Get config
    fpath = config['fpath']
    incl_measurements = config['incl_measurements']
    rename_dict = config['rename_dict']
    save_path = config['save_path']

    physical_measurement_df = pd.read_csv(fpath)
    physical_measurement_df['measurement_date'] = pd.to_datetime(physical_measurement_df['measurement_datetime'].str[0:10])
    physical_measurement_df = physical_measurement_df[['person_id', 'standard_concept_name', 
                                                       'measurement_date', 'value_as_number']].drop_duplicates()
    
    # Only take the necessary measurement types
    physical_measurement_df = physical_measurement_df.loc[physical_measurement_df['standard_concept_name'].isin(incl_measurements)]\
                            .drop_duplicates().dropna()

    # Take the mean if there are multiple measurements on the day
    physical_measurement_df = physical_measurement_df.groupby(
                            ['person_id', 'measurement_date', 'standard_concept_name']).mean().reset_index()


    # Pivot so that the measurements become columns
    physical_measurement_df = physical_measurement_df.loc[physical_measurement_df['standard_concept_name'].isin(incl_measurements)]\
                            .drop_duplicates().dropna()\
                            .pivot(index=['person_id', 'measurement_date'], 
                                   columns='standard_concept_name', 
                                   values='value_as_number')\
                            .reset_index().rename(columns=rename_dict)
    
    # Save the physical measurement df
    physical_measurement_df.to_csv(save_path, index=False)

    return physical_measurement_df


def main():

    # fetch the configuration
    with open('conf/dp_covariates.yaml') as f:
         config = yaml.safe_load(f)
    process_physical_measurement(config['physical_measurement_df'])
    process_demo(config['person_df'])
    process_osa_device(config['osa_device_df'])

if __name__ == '__main__':

    main()

