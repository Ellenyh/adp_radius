import numpy as np

env_params = {
't_initial': 0,
't_end' : 86400,
'delta_t' : 60,
'vehicle_speed' : 6,
'speed_reduce_para' : 2,
'maximum_wait_time_mean' : np.inf,
'maximum_wait_time_std' : 0,
'max_idle_time' : 360,
'whole_area_side_length': 8413,
'whole_area_side_width': 2738,
'block_area': 1 * (10 **6),
'beta': 0.04,
'original_unit_arrival_rate' : 160 / 3600 / 10**6,
'driver_file_name' : 'df_driver_info_94',
'radius' : [2,3,4,5,6] # **
}
