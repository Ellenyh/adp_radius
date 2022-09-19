import copy

import numpy as np
import pandas as pd
from copy import deepcopy
import random
from lyh_config import *
from simulator_pattern import *
from numpy.random import choice
import timeit
import time
import math
from lyh_utilities import *
pd.set_option('expand_frame_repr', False)
class Simulator:
    def __init__(self, **kwargs):
        # basic parameters:
        self.t_initial = kwargs['t_initial']
        self.t_end = kwargs['t_end']
        self.delta_t = kwargs['delta_t']
        self.vehicle_speed = kwargs['vehicle_speed']
        self.speed_reduce_para = kwargs['speed_reduce_para']

        # wait cancel
        self.maximum_wait_time_mean = kwargs.pop('maximum_wait_time_mean', 120)
        self.maximum_wait_time_std = kwargs.pop('maximum_wait_time_std', 0)

        # max idle time before cruising
        self.max_idle_time = kwargs['max_idle_time']

        # grid system
        self.whole_area_side_length = kwargs['whole_area_side_length']
        self.whole_area_side_width = kwargs['whole_area_side_width']
        self.block_area = kwargs['block_area']
        grid_system_params = {
            'range_lng' : np.array([0, self.whole_area_side_width, self.whole_area_side_width, 0]),
            'range_lat' : np.array([0, 0, self.whole_area_side_length, self.whole_area_side_length]),
            'block_area' : self.block_area
        }
        self.GS = GridSystem(**grid_system_params)
        self.GS.construct_grid_system()
        self.num_zone = self.GS.get_basics()

        # pattern
        self.beta = kwargs['beta']
        self.original_unit_arrival_rate = kwargs['original_unit_arrival_rate']
        self.unit_arrival_rate = self.beta * self.original_unit_arrival_rate
        self.arrival_rate = self.unit_arrival_rate * self.GS.total_area * self.delta_t
        self.driver_file_name = kwargs['driver_file_name']
        #pattern_params = {'driver_file_name' : self.driver_file_name}
        #pattern = SimulatorPattern(**pattern_params)

        # get steps
        self.finish_run_step = int((self.t_end - self.t_initial) // self.delta_t)

        # request tables
        self.request_columns = ['order_id', 'trip_time', 'origin_lng', 'origin_lat', 'dest_lng', 'dest_lat',
                                'immediate_reward', 'dest_grid_id', 't_start', 't_matched', 'pickup_time',
                                'wait_time', 't_end', 'status', 'driver_id', 'maximum_wait_time', 'cancel_prob',
                                'pickup_distance', 'weight']
        self.wait_requests = None
        self.matched_requests = None
        self.requests_final_records = None

        # driver tables
        self.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time']
        self.driver_table = None

        # generate driver info
        #self.sampled_driver_info = pattern.driver_info
        total_area_side_length = 20000
        num_drivers = 200
        print('num of drivers',num_drivers)
        data = np.zeros([num_drivers, len(self.driver_columns)])
        df_driver_info = pd.DataFrame(data, columns=self.driver_columns)
        driver_id = np.array([str(i) for i in range(num_drivers)])
        start_time = self.t_initial
        end_time = self.t_end
        grid_id_array = np.zeros(num_drivers).astype(int)
        for i in range(num_drivers):
            grid_id_index = i % self.num_zone
            grid_id_array[i] = grid_id_index
        lng_array = self.GS.df_zone_info.loc[grid_id_array, 'centroid_lng'].values
        lat_array = self.GS.df_zone_info.loc[grid_id_array, 'centroid_lat'].values
        df_driver_info['driver_id'] = driver_id
        df_driver_info['start_time'] = start_time
        df_driver_info['end_time'] = end_time
        df_driver_info['lng'] = lng_array
        df_driver_info['lat'] = lat_array
        df_driver_info['grid_id'] = grid_id_array
        # pickle.dump(df_driver_info, open(load_path + 'df_driver_info_' + str(num_drivers) + '_' +
        #                                  str(self.GS.total_area / 10 ** 6) + '_' +
        #                                  str(self.GS.avg_block_area / 10 ** 6) +  '.pickle', 'wb'))
        self.sampled_driver_info = df_driver_info
        self.sampled_driver_info['grid_id'] = self.sampled_driver_info['grid_id'].values.astype(int)

        # matching related tables
        self.order_queue_columns = ['order_id', 'origin_lng', 'origin_lat']
        self.driver_queue_columns = ['driver_id', 'lng', 'lat']
        self.order_queue_dict = {}
        self.driver_queue_dict = {}
        self.order_queue_count_array = np.zeros(self.num_zone)
        self.driver_queue_count_array = np.zeros(self.num_zone)

        # record for measurements


    def initial_base_tables(self):
        # driver table
        self.driver_table = sample_all_drivers(self.sampled_driver_info, self.t_initial, self.t_end)
        self.driver_table['target_grid_id'] = self.driver_table['target_grid_id'].values.astype(int)

        # order
        self.wait_requests = pd.DataFrame(columns=self.request_columns)  # state: (wait 0, matched 1, finished 2)
        self.matched_requests = pd.DataFrame(columns=self.request_columns)
        self.order_id_counter = 0
        self.requests_final_records = []

        # time
        self.time = deepcopy(self.t_initial)
        self.current_step = int((self.time - self.t_initial) // self.delta_t)

        #matching related tables
        self.order_queue_dict = {}
        self.driver_queue_dict = {}
        grid_id_list = self.GS.df_zone_info['grid_id'].values.tolist()
        for grid_id in grid_id_list:
            self.order_queue_dict[grid_id] = pd.DataFrame(columns=self.order_queue_columns)
            self.driver_queue_dict[grid_id] = pd.DataFrame(columns=self.driver_queue_columns)
        self.order_queue_count_array = np.zeros(self.num_zone)
        self.driver_queue_count_array = np.zeros(self.num_zone)
        for i in range(self.driver_table.shape[0]):
            grid_id = self.driver_table.loc[i, 'grid_id']
            data = self.driver_table.loc[i, ['driver_id', 'lng', 'lat']].values
            queue_length = self.driver_queue_dict[grid_id].shape[0]
            self.driver_queue_dict[grid_id].loc[queue_length, ['driver_id', 'lng', 'lat']] = data
            self.driver_queue_count_array[grid_id] += 1

        # measurements
        self.cumulated_queueing_time_matched = 0
        self.num_for_queueing_time_matched = 0
        self.cumulated_queueing_time_waiting = 0
        self.num_for_queueing_time_waiting = 0
        self.cumulated_pickup_time = 0
        self.num_for_pickup_time = 0


    def reset(self):
        self.initial_base_tables()

    def update_info_after_matching_multi_process(self, matched_pair_actual_indexes, new_matched_pickup_dis):
        new_matched_requests = pd.DataFrame([], columns=self.request_columns)
        update_wait_requests = pd.DataFrame([], columns=self.request_columns)

        matched_pair_index_df = pd.DataFrame(matched_pair_actual_indexes,
                                             columns=['order_id', 'driver_id'])
        matched_pair_index_df['pickup_distance'] = new_matched_pickup_dis
        matched_order_id_list = matched_pair_index_df['order_id'].values.tolist()
        con_matched = self.wait_requests['order_id'].isin(matched_order_id_list)
        con_keep_wait = self.wait_requests['wait_time'] <= self.wait_requests['maximum_wait_time']

        #when the order is matched
        df_matched = self.wait_requests[con_matched].reset_index(drop=True)
        if df_matched.shape[0] > 0:
            idle_driver_table = self.driver_table[(self.driver_table['status'] == 0) | (self.driver_table['status'] == 2)]
            order_array = df_matched['order_id'].values

            # multi process if necessary
            cor_order = []
            cor_driver = []
            for i in range(len(matched_pair_index_df)):
                cor_order.append(np.argwhere(order_array == matched_pair_index_df['order_id'][i])[0][0])
                cor_driver.append(idle_driver_table[idle_driver_table['driver_id'] == matched_pair_index_df['driver_id'][i]].index[0])

            if (len(cor_order) != len(matched_pair_index_df)) or (len(cor_driver) != len(matched_pair_index_df)):
                raise

            cor_driver = np.array(cor_driver)
            df_matched = df_matched.iloc[cor_order, :]

            #decide whether cancelled
            # currently no cancellation
            cancel_prob = -1 + np.zeros(len(matched_pair_index_df))
            prob_array = np.random.rand(len(cancel_prob))
            con_remain = prob_array >= cancel_prob

            # order after cancelled
            update_wait_requests = df_matched[~con_remain]

            # driver after cancelled
            #self.driver_table.loc[cor_driver[~con_remain], ['status', 'remaining_time', 'total_idle_time']] = 0

            # order not cancelled
            new_matched_requests = df_matched[con_remain]
            new_matched_requests['t_matched'] = self.time
            new_matched_requests['pickup_distance'] = matched_pair_index_df[con_remain]['pickup_distance'].values
            new_matched_requests['pickup_time'] = new_matched_requests['pickup_distance'].values / self.vehicle_speed
            new_matched_requests['t_end'] = self.time + new_matched_requests['pickup_time'].values + new_matched_requests['trip_time'].values
            new_matched_requests['status'] = 1
            new_matched_requests['driver_id'] = matched_pair_index_df[con_remain]['driver_id'].values

            # driver not cancelled
            self.driver_table.loc[cor_driver[con_remain], 'status'] = 1
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lng'] = new_matched_requests['dest_lng'].values
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lat'] = new_matched_requests['dest_lat'].values
            self.driver_table.loc[cor_driver[con_remain], 'target_grid_id'] = new_matched_requests['dest_grid_id'].values
            self.driver_table.loc[cor_driver[con_remain], 'remaining_time'] = new_matched_requests['t_end'].values \
                                                                              - new_matched_requests['t_matched'].values
            self.driver_table.loc[cor_driver[con_remain], 'matched_order_id'] = new_matched_requests['order_id'].values
            self.driver_table.loc[cor_driver[con_remain], 'total_idle_time'] = 0

            self.requests_final_records += new_matched_requests.values.tolist()

        # when the order is not matched
        update_wait_requests = pd.concat([update_wait_requests, self.wait_requests[~con_matched & con_keep_wait]],axis=0)
        self.requests_final_records += self.wait_requests[~con_matched & ~con_keep_wait].values.tolist()

        # statistics for measurements
        self.cumulated_queueing_time_matched += np.sum(new_matched_requests['wait_time'])
        self.num_for_queueing_time_matched += len(new_matched_requests)
        self.cumulated_queueing_time_waiting = np.sum(update_wait_requests['wait_time'])
        self.num_for_queueing_time_waiting = len(update_wait_requests)
        self.cumulated_pickup_time += np.sum(new_matched_requests['pickup_time'])
        self.num_for_pickup_time += len(new_matched_requests)


        return new_matched_requests, update_wait_requests


    def generate_new_orders(self,df,cur_step):
       # p = np.exp(-self.arrival_rate) * self.arrival_rate
       # flag = np.random.choice(2, 1, p=[1-p, p])
       flag = 1
       for order_od in df[str(cur_step)]:
           o = np.array(order_od[0])
           d = np.array(order_od[1])
           o_grid_id = self.GS.generate_points(o)
           d_grid_id = self.GS.generate_points(d)

           order_id = str(self.order_id_counter)
           trip_dis = line_distance_on_plane_array(o.reshape((1,2)),d.reshape((1,2)))
           trip_time = trip_dis / self.vehicle_speed * self.speed_reduce_para
           new_order = [order_id, trip_time, o[0], o[1], d[0], d[1], 1, d_grid_id,
                        self.time, 0, 0, 0, 0, 0, 'None', self.maximum_wait_time_mean, 0, 0, 1]
           self.wait_requests.loc[self.wait_requests.shape[0]] = new_order
           order_queue_length = self.order_queue_count_array[o_grid_id]
           self.order_queue_dict[o_grid_id].loc[order_queue_length] = [order_id, o[0], o[1]]
           self.order_queue_count_array[o_grid_id] += 1
           self.order_id_counter += 1
       return


    def cruising(self):
        loc_long_idle = (self.driver_table['total_idle_time'] >= self.max_idle_time) & (self.driver_table['status'] == 0)
        grid_id_array = self.driver_table.loc[loc_long_idle, 'grid_id'].values
        if grid_id_array.shape[0] > 0:
            grid_id_indices = grid_id_array.astype(int)
            adj_mat_by_grid = self.GS.adj_mat[grid_id_indices]
            cruising_prob_by_grid = adj_mat_by_grid / np.sum(adj_mat_by_grid, axis=1)[:,None]
            c = cruising_prob_by_grid.cumsum(axis=1)
            u = np.random.rand(len(c), 1)
            choices = (u < c).argmax(axis=1)
            range_array = self.GS.df_zone_info.loc[choices, ['lng_0', 'lat_0', 'lng_2', 'lat_2']].values
            target_lng_array, target_lat_array = generate_points_for_squares(range_array)
            target_grid_array = self.GS.df_zone_info.loc[choices, 'grid_id'].values
            self.driver_table.loc[loc_long_idle, 'status'] = 2
            self.driver_table.loc[loc_long_idle, 'target_loc_lng'] = target_lng_array
            self.driver_table.loc[loc_long_idle, 'target_loc_lat'] = target_lat_array
            self.driver_table.loc[loc_long_idle, 'target_grid_id'] = target_grid_array
            self.driver_table.loc[loc_long_idle, 'total_idle_time'] = 0
            cruising_distance = \
                line_distance_on_plane_array(self.driver_table.loc[loc_long_idle, ['lng', 'lat']].values,
                                             self.driver_table.loc[loc_long_idle, ['target_loc_lng', 'target_loc_lat']].values)
            self.driver_table.loc[loc_long_idle, 'remaining_time'] = cruising_distance / self.vehicle_speed


    def update_state(self):
        # update next state
        # update driver information
        self.driver_table['remaining_time'] = self.driver_table['remaining_time'].values - self.delta_t
        loc_negative_time = self.driver_table['remaining_time'] <= 0
        loc_idle = self.driver_table['status'] == 0
        loc_on_trip = self.driver_table['status'] == 1
        loc_cruising = self.driver_table['status'] == 2

        #for all drivers
        self.driver_table.loc[loc_negative_time, 'remaining_time'] = 0

        # for cruising and delivery dirvers
        loc_delivery_finished = loc_negative_time & loc_on_trip
        loc_cruising_finished = loc_negative_time & loc_cruising
        loc_task_finished = loc_delivery_finished | loc_cruising_finished
        cruising_grid_id_array = self.driver_table.loc[loc_cruising_finished, 'grid_id'].values

        self.driver_table.loc[loc_task_finished, 'lng'] = self.driver_table.loc[
            loc_task_finished, 'target_loc_lng'].values
        self.driver_table.loc[loc_task_finished, 'lat'] = self.driver_table.loc[
            loc_task_finished, 'target_loc_lat'].values
        self.driver_table.loc[loc_task_finished, 'grid_id'] = self.driver_table.loc[
            loc_task_finished, 'target_grid_id'].values.astype(int)
        self.driver_table.loc[loc_task_finished, 'status'] = 0

        cruising_driver_id_array = self.driver_table.loc[loc_cruising_finished, 'driver_id'].values
        unique_cruising_grid_id_array = np.unique(cruising_grid_id_array)

        for i in range(len(cruising_grid_id_array)):
            grid_id = cruising_grid_id_array[i]
            con = self.driver_queue_dict[grid_id]['driver_id'] == cruising_driver_id_array[i]
            index = self.driver_queue_dict[grid_id][con].index

            if len(index) == 0:
                raise

            self.driver_queue_dict[grid_id] = self.driver_queue_dict[grid_id].drop(index)
            self.driver_queue_count_array[grid_id] -= 1

        for grid_id in unique_cruising_grid_id_array:
            self.driver_queue_dict[grid_id] = self.driver_queue_dict[grid_id].reset_index(drop=True)


        grid_id_array = self.driver_table.loc[loc_task_finished, 'grid_id'].values
        driver_info_array = self.driver_table.loc[loc_task_finished, ['driver_id', 'lng', 'lat']].values
        for i in range(len(grid_id_array)):
            grid_id = grid_id_array[i]
            self.driver_queue_dict[grid_id].loc[self.driver_queue_count_array[grid_id]] = driver_info_array[i, :]
            self.driver_queue_count_array[grid_id] += 1

        # for idle drivers
        self.driver_table.loc[loc_idle, 'total_idle_time'] += self.delta_t

        # for delivery drivers
        self.driver_table.loc[loc_delivery_finished, 'matched_order_id'] = 'None'

        # for cruising drivers

        # wait list update
        self.wait_requests['wait_time'] += self.delta_t

        return

    def update_time(self):
        # time counter
        self.time += self.delta_t
        self.current_step = int((self.time - self.t_initial) // self.delta_t)

        return

    def grid_distance(self):
        self.GS.get_grid_distance()


    def step(self, radius, grid_distance,cur_step,df):
        # print(radius)
        done = 0
        r = 0

        for time_step in range(1):

            # Step 1: block matching

            new_matched_pairs, new_matched_pickup_dis, self.order_queue_dict, self.driver_queue_dict, self.order_queue_count_array, \
            self.driver_queue_count_array, time_match, time_match1 \
                = block_matching(radius, grid_distance, self.order_queue_dict, self.driver_queue_dict,
                                 self.order_queue_count_array, self.driver_queue_count_array)
            # delay_time = 0
            # for id_od in new_matched_pairs:
            #     delay_time += self.wait_requests.loc[self.wait_requests['order_id'] == id_od[0]]['wait_time'].values[0]
            # Step 2: reaction after matching
            df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(new_matched_pairs, new_matched_pickup_dis)
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
            self.matched_requests = self.matched_requests.reset_index(drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)
            r = -(df_new_matched_requests['wait_time'].sum() + df_new_matched_requests['pickup_time'].sum())
            if cur_step == 0:
                r = 0
            # Step 3: generate new orders
            # if cur_step != 0 and cur_step % 3600 == 0:
            if cur_step == 86400:
                done = 1
                r = -self.wait_requests['wait_time'].sum()
                break

            self.generate_new_orders(df,cur_step)

            # Step 4: cruising
            self.cruising()

            # Step 5: update next state
            self.update_state()

            # Step 6: update time
            self.update_time()

        state = copy.deepcopy([self.order_queue_count_array,self.driver_queue_count_array,radius])
        return state, r, done
