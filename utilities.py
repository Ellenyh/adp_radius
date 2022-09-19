import numpy as np
import pandas as pd
from copy import deepcopy
from lyh_config import *
from path import *
import pickle
import time

def line_distance_on_plane_array(coord_1_array, coord_2_array, detour_ratio = 1.27):
    # caculate line distances on a plane
    coord_1 = coord_1_array.astype(float)
    coord_2 = coord_2_array.astype(float)
    dlon = coord_2_array[:, 0] - coord_1_array[:, 0]
    dlat = coord_2_array[:, 1] - coord_1_array[:, 1]
    distance_array = (dlon**2 + dlat**2) ** 0.5 * detour_ratio
    return distance_array

class GridSystem:
    def __init__(self, **kwargs):
        #read parameters
        # range_lng and range_lat are two arrays containing lng and lat of the four vertices of the ranctangular range
        self.range_lng = kwargs['range_lng']
        self.range_lat = kwargs['range_lat']
        self.avg_block_area = kwargs['block_area']
        self.area_side_length = self.range_lat[3] - self.range_lat[0]
        self.area_side_width = self.range_lng[1] - self.range_lng[0]
        self.total_area = self.area_side_length * self.area_side_width
        self.avg_block_side_length = self.avg_block_area ** 0.5
        self.num_meshes_length = int(np.ceil(self.area_side_length / self.avg_block_side_length))
        self.num_meshes_width = int(np.ceil(self.area_side_width / self.avg_block_side_length))
        self.num_grid = int(self.num_meshes_width * self.num_meshes_length)


    def construct_grid_system(self):
        print('construct grid system')

        # construct information for each zone
        columns = ['grid_id', 'lng_0', 'lat_0', 'lng_1', 'lat_1', 'lng_2', 'lat_2', 'lng_3', 'lat_3',
                   'centroid_lng', 'centroid_lat']
        df_zone_info = pd.DataFrame(data=np.zeros([self.num_grid, len(columns)]), columns=columns)
        length_for_meshes = self.avg_block_side_length + np.zeros(self.num_meshes_length)
        length_for_meshes[-1] = self.area_side_length - np.sum(length_for_meshes[:-1])
        width_for_meshes = self.avg_block_side_length + np.zeros(self.num_meshes_width)
        width_for_meshes[-1] = self.area_side_width - np.sum(width_for_meshes[:-1])
        df_zone_info['grid_id'] = np.array(range(self.num_grid))
        zone_data = np.zeros([self.num_grid, 8])

        for i in range(self.num_grid):
            m = i % self.num_meshes_width # vertical
            n = i // self.num_meshes_width # horizontal
            zone_data[i, 0] = m * self.avg_block_side_length
            zone_data[i, 1] = n * self.avg_block_side_length
            zone_data[i, 2] = zone_data[i, 0] + length_for_meshes[m]
            zone_data[i, 3] = zone_data[i, 1]
            zone_data[i, 4] = zone_data[i, 2]
            zone_data[i, 5] = zone_data[i, 1] + length_for_meshes[n]
            zone_data[i, 6] = zone_data[i, 0]
            zone_data[i, 7] = zone_data[i, 5]
        df_zone_info.iloc[:, 1:9] = zone_data
        df_zone_info.loc[:, 'centroid_lng'] = (zone_data[:, 0] + zone_data[:, 2]) / 2
        df_zone_info.loc[:, 'centroid_lat'] = (zone_data[:, 1] + zone_data[:, 5]) / 2
        self.df_zone_info = df_zone_info
        print(df_zone_info)

        # construct adjacent matrix
        adj_mat = np.zeros([self.num_grid, self.num_grid])
        zone_mat = np.flipud(np.array(range(self.num_grid)).reshape([self.num_meshes_width, self.num_meshes_length]).T)
        print(zone_mat)
        for i in range(self.num_meshes_length):
            for j in range(self.num_meshes_width):
                grid_id = zone_mat[i, j]
                if j - 1 >= 0:
                    adj_mat[grid_id, zone_mat[i, j-1]] = 1
                if i + 1 <= self.num_meshes_length - 1:
                    adj_mat[grid_id, zone_mat[i+1, j]] = 1
                if j + 1 <= self.num_meshes_width - 1:
                    adj_mat[grid_id, zone_mat[i, j+1]] = 1
                if i - 1 >= 0:
                    adj_mat[grid_id, zone_mat[i-1, j]] = 1
        self.adj_mat = adj_mat

        # # store the matrices
        # pickle.dump(df_zone_info,
        #             open(load_path + 'df_zone_info_' + str(self.total_area/10**6) + '_' + str(self.avg_block_area/10**6) + '_' + '.pickle', 'wb'))
        # pickle.dump(adj_mat,
        #             open(load_path + 'adj_mat_' + str(self.total_area/10**6) + '_' + str(self.avg_block_area/10**6) + '_' + '.pickle', 'wb'))

    def get_grid_distance(self):
        self.grid_distance = np.empty([self.num_grid, self.num_grid])
        for i in range(self.num_grid):
            position_i = np.array([self.df_zone_info.loc[i, 'centroid_lng'], self.df_zone_info.loc[i, 'centroid_lat']])
            position_i = np.tile(position_i,[self.num_grid,1])
            position_other = self.df_zone_info.loc[:,['centroid_lng','centroid_lat']].values
            self.grid_distance[i] = line_distance_on_plane_array(position_i, position_other)

    # def load_from_existed_grid_system(self):
    #     self.df_zone_info = pickle.load(
    #         open(load_path + 'df_zone_info_' + str(self.total_area) + '_' + str(self.avg_block_area) + '_' + '.pickle', 'rb'))
    #     self.adj_mat = pickle.load(
    #         open(load_path + 'adj_mat_' + str(self.total_area) + '_' + str(self.avg_block_area) + '_' + '.pickle', 'rb'))

    def get_basics(self):
        # output: basic information about the grid network
        return self.num_grid

    def generate_points(self, x):
        """
        生成demand需要基于data
        :return:
        """
        m_array = (x[0] // self.avg_block_side_length).astype(int)
        n_array = (x[1] // self.avg_block_side_length).astype(int)
        grid_id = n_array * self.num_meshes_width + m_array
        return grid_id

def generate_points_for_squares(range_array):
    """
    cursing用到的 暂时不管
    :param range_array:
    :return:
    """
    # range array [[lng 0, lat 0, lng 2, lat 2], ...]
    lng_array = np.random.uniform(low=range_array[:, 0], high=range_array[:, 2], size=range_array.shape[0])
    lat_array = np.random.uniform(low=range_array[:, 1], high=range_array[:, 3], size=range_array.shape[0])
    return lng_array, lat_array

def sample_all_drivers(driver_info, t_initial, t_end, driver_sample_ratio=1):
    new_driver_info = deepcopy(driver_info)
    sampled_driver_info = new_driver_info
    sampled_driver_info['status'] = 0
    sampled_driver_info['target_loc_lng'] = sampled_driver_info['lng']
    sampled_driver_info['target_loc_lat'] = sampled_driver_info['lat']
    sampled_driver_info['target_grid_id'] = sampled_driver_info['grid_id']
    sampled_driver_info['remaining_time'] = 0
    sampled_driver_info['matched_order_id'] = 'None'
    sampled_driver_info['total_idle_time'] = 0

    return sampled_driver_info

def block_matching(radius, grid_distance, order_queue_dict, driver_queue_dict, order_queue_count_array, driver_queue_count_array):
    """
    根据distance来匹配，从最短到最长
    """
    time_match1 = 0
    time_match = 0
    start_time = time.time()
    new_matched_pairs = []
    new_matched_pickup_dis = []
    # matching
    grid_index_to_match = np.where(order_queue_count_array > 0)[0]
    search_area_dict = {}
    for grid_index in grid_index_to_match:
        grid_radius = radius[grid_index]
        search_area = np.where(grid_distance[grid_index] < grid_radius)[0]
        search_area_dict[grid_index] = search_area
    # for drivers
    all_possible_pairs_column = ['order_id','driver_id','distance','grid_id','grid_driver_id']
    all_possible_pairs = pd.DataFrame(columns=all_possible_pairs_column)

    # generate all possible pairs
    for grid_index in grid_index_to_match:
        pairs_order_search_area = search_area_dict[grid_index]
        pairs_driver = pd.concat(driver_queue_dict[i] for i in pairs_order_search_area).values
        pairs_driver_grid_id = np.array\
            ([[val for val in pairs_order_search_area for i in range(driver_queue_dict[val].shape[0])]]).T
        pairs_driver = np.hstack((pairs_driver,pairs_driver_grid_id))
        len_driver = int(pairs_driver.shape[0])
        len_order = int(order_queue_count_array[grid_index])
        pairs_order = np.repeat(order_queue_dict[grid_index].values, len_driver, axis=0)
        pairs_driver = np.tile(pairs_driver,(len_order,1))
        pairs_gird_id = np.array([grid_index] * pairs_order.shape[0])
        pairs_distance = line_distance_on_plane_array(pairs_order[:,1:3], pairs_driver[:,1:3])
        all_possible_pairs_grid_pd = pd.DataFrame\
            (np.vstack((pairs_order[:,0], pairs_driver[:,0], pairs_distance, pairs_gird_id, pairs_driver[:,3])).T, columns=all_possible_pairs_column)
        all_possible_pairs = all_possible_pairs.append(all_possible_pairs_grid_pd, ignore_index = True)
    all_possible_pairs.sort_values("distance",inplace=True)
    all_possible_pairs = all_possible_pairs.reset_index(drop=True)

    # update related tables
    while len(all_possible_pairs) > 0:
        pair_order_id = all_possible_pairs.loc[0,['order_id']].values[0]
        pair_driver_id = all_possible_pairs.loc[0,['driver_id']].values[0]
        pair_grid_driver_id = all_possible_pairs.loc[0,['grid_driver_id']].values[0]
        pair_distance = all_possible_pairs.loc[0,['distance']].values[0]
        pair_grid_id = all_possible_pairs.loc[0,['grid_id']].values[0]
        new_matched_pairs.append([pair_order_id, pair_driver_id])
        new_matched_pickup_dis.append(pair_distance)

        # delete the matched order and driver
        all_possible_pairs = all_possible_pairs.drop(all_possible_pairs[all_possible_pairs['order_id'] == str(pair_order_id)].index)
        all_possible_pairs = all_possible_pairs.drop(all_possible_pairs[all_possible_pairs['driver_id'] == str(pair_driver_id)].index).reset_index(drop=True)

        # update table infotmation
        order_queue_dict[pair_grid_id] = order_queue_dict[pair_grid_id].drop\
            (order_queue_dict[pair_grid_id][order_queue_dict[pair_grid_id]['order_id'] == pair_order_id].index).reset_index(drop=True)

        driver_queue_dict[pair_grid_driver_id] = driver_queue_dict[pair_grid_driver_id].drop\
            (driver_queue_dict[pair_grid_driver_id][driver_queue_dict[pair_grid_driver_id]['driver_id'] == pair_driver_id].index).reset_index(drop=True)
        order_queue_count_array[pair_grid_id] = order_queue_count_array[pair_grid_id] - 1
        driver_queue_count_array[pair_grid_driver_id] = driver_queue_count_array[pair_grid_driver_id] - 1
    end_time = time.time()
    time_match = end_time - start_time
    time_match1 = time_match

    return new_matched_pairs, new_matched_pickup_dis, order_queue_dict, driver_queue_dict, order_queue_count_array, driver_queue_count_array, time_match, time_match1


# def block_matching(grid_id_array, order_queue_dict, driver_queue_dict, order_queue_count_array, driver_queue_count_array):
#     time_match1 = 0
#     start_time = time.time()
#     # matching by queues for each block respectively
#     new_matched_pairs = []
#     new_matched_pickup_dis = []
#
#     grid_index_to_match = np.where((order_queue_count_array > 0) & (driver_queue_count_array > 0))[0]
#     for grid_index in grid_index_to_match:
#         grid_id = grid_id_array[grid_index]
#         if order_queue_count_array[grid_index] >= driver_queue_count_array[grid_index]:
#             # generate matched pairs
#             start_time1 = time.time()
#             driver_id_array = driver_queue_dict[grid_id].loc[:, 'driver_id'].values
#             driver_coord_array = driver_queue_dict[grid_id].loc[:, ['lng', 'lat']].values
#             driver_queue_length = len(driver_id_array)
#             order_id_array = order_queue_dict[grid_id].loc[:(driver_queue_length-1), 'order_id'].values
#             order_coord_array = order_queue_dict[grid_id].loc[:(driver_queue_length-1), ['origin_lng', 'origin_lat']].values
#             pickup_dis_array = line_distance_on_plane_array(driver_coord_array, order_coord_array)
#             pairs = np.vstack([order_id_array, driver_id_array]).T
#
#             new_matched_pairs.append(pairs)
#             new_matched_pickup_dis.append(pickup_dis_array)
#
#             end_time1 = time.time()
#             time_match1 += end_time1 - start_time1
#
#             #update matching related tables
#             order_queue_count_array[grid_index] -= driver_queue_length
#             driver_queue_count_array[grid_index] = 0
#             order_queue_dict[grid_id] = order_queue_dict[grid_id].iloc[driver_queue_length:, :].reset_index(drop=True)
#             driver_queue_dict[grid_id] = pd.DataFrame(columns=driver_queue_dict[grid_id].columns)
#
#         else:
#             start_time1 = time.time()
#             # generate matched pairs
#             order_id_array = order_queue_dict[grid_id].loc[:, 'order_id'].values
#             order_coord_array = order_queue_dict[grid_id].loc[:, ['origin_lng', 'origin_lat']].values
#             order_queue_length = len(order_id_array)
#
#             # matching for the last order
#             remain_driver_coord_array = driver_queue_dict[grid_id].loc[(order_queue_length-1):, ['lng', 'lat']].values
#             last_order_coord_array = np.tile(order_coord_array[-1, :], remain_driver_coord_array.shape[0]).reshape([-1, 2])
#             last_order_pickup_dis_array = line_distance_on_plane_array(remain_driver_coord_array, last_order_coord_array)
#             remain_driver_index = np.argmin(last_order_pickup_dis_array)
#             last_pair = np.array([
#                 order_id_array[-1],
#                 driver_queue_dict[grid_id]['driver_id'][order_queue_length - 1 + remain_driver_index],
#             ])
#             last_pickup_dis = last_order_pickup_dis_array[remain_driver_index]
#             driver_queue_dict[grid_id] = driver_queue_dict[grid_id].drop([order_queue_length - 1 + remain_driver_index])
#
#             # matching for orders except the last one
#             driver_id_array = driver_queue_dict[grid_id].loc[:(order_queue_length-2), 'driver_id'].values
#             driver_coord_array = driver_queue_dict[grid_id].loc[:(order_queue_length-2), ['lng', 'lat']].values
#             pickup_dis_array = line_distance_on_plane_array(driver_coord_array, order_coord_array[:-1, :])
#             pairs = np.vstack([order_id_array[:-1], driver_id_array]).T
#             pairs = np.vstack([pairs, last_pair])
#             pickup_dis_array = np.concatenate([pickup_dis_array, np.array([last_pickup_dis])])
#             new_matched_pairs.append(pairs)
#             new_matched_pickup_dis.append(pickup_dis_array)
#
#             end_time1 = time.time()
#             time_match1 += end_time1 - start_time1
#
#             #update matching related tables
#             order_queue_count_array[grid_index] = 0
#             driver_queue_count_array[grid_index] -= order_queue_length
#             order_queue_dict[grid_id] = pd.DataFrame(columns=order_queue_dict[grid_id].columns)
#             driver_queue_dict[grid_id] = driver_queue_dict[grid_id].iloc[(order_queue_length-1):, :].reset_index(drop=True)
#
#     if len(new_matched_pairs) > 0:
#         new_matched_pairs = np.vstack(new_matched_pairs)
#         new_matched_pickup_dis = np.concatenate(new_matched_pickup_dis)
#
#     end_time = time.time()
#     time_match = end_time - start_time
#
    # return new_matched_pairs, new_matched_pickup_dis, order_queue_dict, driver_queue_dict, order_queue_count_array, driver_queue_count_array, time_match, time_match1




