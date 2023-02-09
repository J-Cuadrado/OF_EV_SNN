import torch

from data.file_generator import generate_files

flow_sequences = ['zurich_city_09_a',
                  'zurich_city_07_a',
                  'zurich_city_02_c',
                  'zurich_city_11_b',
                  'thun_00_a',
                  'zurich_city_02_d',
                  'zurich_city_11_c',
                  'zurich_city_03_a',
                  'zurich_city_10_a',
                  'zurich_city_05_b',
                  'zurich_city_08_a',
                  'zurich_city_01_a',
                  'zurich_city_10_b',
                  'zurich_city_02_e',
                  'zurich_city_05_a',
                  'zurich_city_06_a',
                  'zurich_city_11_a',
                  'zurich_city_02_a']


if __name__=='__main__':

        for sequence in flow_sequences:

            generate_files(root = '../data/dataset', sequence = sequence, num_frames_per_ts = 11)
