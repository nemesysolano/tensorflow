import tensorflow as tf
import pandas as pd
import numpy as np
import os
from multiprocessing import cpu_count, Pool
from cymath import ema_value, atan_circular,zoomed_mantissa
not_zero = lambda x: np.abs(x) > 0.0005
double_pi = np.pi * 2

def roll_array(array, count = 1):
    count = count % len(array)
    for _ in range(count):
        last_element = array[-1]
        for i in range(len(array)-1, 0, -1):
            array[i] = array[i-1]
        array[0] = last_element
    return array

def rescale(value):
    # original range
    original_min = -1
    original_max = 1
    value = -1 if value < -1 else ( 1 if value > 1 else value)
    # new range
    new_min = 0
    new_max = 1

    # rescale the value
    rescaled_value = new_min + (value - original_min) * (new_max - new_min) / (original_max - original_min)

    return rescaled_value


def load_dataset(file_path):
    source_dataset = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')
    smooth = np.array((0,0,0,0,0,0,0)).astype(np.float32)
    detrender = np.array((0,0,0,0,0,0,0)).astype(np.float32)
    period = np.array((0,0,0,0,0,0,0)).astype(np.float32)
    q1 = np.array((0,0,0,0,0,0,0)).astype(np.float32)
    i1 = np.array((0,0,0,0,0,0,0)).astype(np.float32)
    ji = np.array((0,0,0,0,0,0,0)).astype(np.float32)
    jq = np.array((0,0,0,0,0,0,0)).astype(np.float32)
    i2 = np.array((0,0,0,0,0,0,0)).astype(np.float32)
    q2 = np.array((0,0,0,0,0,0,0)).astype(np.float32)    
    im = np.array((0,0,0,0,0,0,0)).astype(np.float32)    
    re = np.array((0,0,0,0,0,0,0)).astype(np.float32)    
    slow_ma = np.array((0,0,0,0,0,0,0)).astype(np.float32)    
    fast_ma = np.array((0,0,0,0,0,0,0)).astype(np.float32)    
    smooth_period = np.array((0,0,0,0,0,0,0)).astype(np.float32)    
    phase = np.array((0,0,0,0,0,0,0)).astype(np.float32)    
    dataset_index = source_dataset.index
    fast_limit = 0.50
    slow_limit = 0.05
    mesa = []
    direction = []
    mesa_values = np.array((0, 0))

    price0, price1, price2, price3 = (0, 0, 0, 0)
    source_dataset.rename(columns=lambda x: x.strip() if isinstance(x, str) else x, inplace=True)

    rollable_arrays =         (
            smooth,
            detrender,
            period,
            q1 ,
            i1,
            ji,
            jq,
            i2,
            q2,
            im , 
            re ,
            slow_ma ,
            fast_ma ,
            smooth_period,
            phase,     
        )
    for index in range(0, len(dataset_index)):
        for arr in rollable_arrays:
            roll_array(arr)

        price0 = source_dataset.iloc[index]['close']
        
        if index == 0:
            fast_ma[0] = price0
            slow_ma[0] = price0
        elif index < 4:
            fast_ma[0] = ema_value(fast_limit, price0, fast_ma[1])
            slow_ma[0] = ema_value(slow_limit, price0, slow_ma[1])
        else:
            price1 = source_dataset.iloc[index-1]['close']
            price2 = source_dataset.iloc[index-2]['close']
            price3 = source_dataset.iloc[index-3]['close']

            smooth[0] = (4 * price0 + 3 * price1 + 2 * price2 + price3) / 10.0
            detrender[0] = (.0962*smooth[0] + .5769*smooth[2] - .5769*smooth[4] - .0962*smooth[6]) * (.075*period[1] + .54)

            q1[0] = (.0962*detrender[0] + .5769*detrender[2] - .5769*detrender[4] - .0962*detrender[6]) * (.075*period[1] + .54)
            i1[0] = detrender[3]

            ji[0] = (.0962*i1[0] + .5769*i1[2] - .5769*i1[4] - .0962*i1[6]) * (.075 * period[1] + 0.54)
            jq[0] = (.0962*q1[0] + .5769*q1[2] - .5769*q1[4] - .0962*q1[6]) * (.075 * period[1] + 0.54)

            i2[0] = i1[0] - jq[0]
            q2[0] = q1[0] + ji[0]

            i2[0] = .2 * i2[0] + .8 * i2[1]
            q2[0] = .2 * q2[0] + .8 * q2[1]

            re[0] = i2[0] * i2[1] + q2[0] * q2[1]
            im[0] = i2[0] * q2[1] - q2[0] * i2[1]

            re[0] = 0.2 * re[0] + .8 * re[1]
            im[0] = 0.2 * im[0] + .8 * im[1]

            if (im[0] != 0.0 and re[0] != 0.0):
                period[0] = double_pi / atan_circular(im[0] , re[0])

            if (period[0] > 1.5 * period[1]): 
                period[0] = 1.5 * period[1]

            if (period[0] < 0.67 * period[1]):
                period[0] = 0.67 * period[1]

            if (period[0] < 6.): 
                period[0] = 6.

            if (period[0] > 50.):
                period[0] = 50.
            period[0] = .2*period[0] + .8*period[1]
            smooth_period[0] = .33*period[0] + .67*smooth_period[1]

            if (i1[0] != 0.):
                phase[0] = atan_circular(q1[0] , i1[0])
            else:
                phase[0] = 0

            delta_phase = np.max((1., phase[1] - phase[0]))
            alpha =  np.max((slow_limit, fast_limit / delta_phase))

            slow_ma[0] = alpha * price0 + (1. - alpha) * slow_ma[1]
            fast_ma[0] = .5 * alpha * slow_ma[0] + (1. - .5 * alpha) * fast_ma[1]

        
        zoomed_mantissa(slow_ma[0], fast_ma[0], 100000, mesa_values)
        direction.append(
            -1 if (fast_ma[0] - slow_ma[0] > 0 and fast_ma[1] - slow_ma[1] < 0) else (1 if (fast_ma[0] - slow_ma[0] < 0 and fast_ma[1] - slow_ma[1] > 0) else 0)
        )
        
        if index > 0:
            mesa.append(rescale((mesa_values[0]-mesa_values[1]) / ((mesa_values[0] + mesa_values[1]) / 2)))
        else:
            mesa.append(0)
            
    
    source_dataset['mesa'] = mesa
    source_dataset['direction'] = direction
    return source_dataset[datasets['normalized_tick_volume'] > 0]


def load_datasets(directory):
    pool = Pool(cpu_count() - 1)
    files = tuple([f"{directory}/{file_name}" for file_name in os.listdir(directory)])
    datasets = pool.map(load_dataset, files)

    return datasets

if __name__ == "__main__":
    datasets = load_datasets("prices")
    print(datasets[0].head(44))