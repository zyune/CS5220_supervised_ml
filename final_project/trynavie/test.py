import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from m4_data import prepare_m4_data


prepare_m4_data(dataset_name='Yearly', directory='.', num_obs=1000)