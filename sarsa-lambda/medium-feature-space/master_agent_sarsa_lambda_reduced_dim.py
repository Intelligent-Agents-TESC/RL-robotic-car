#from numba import jit, njit, prange
import numpy as np
import serial
import random
import math

# servant arduino
serial = serial.Serial('/dev/ttyACM0', 115200, timeout=0.5)

file_path_dict = {
        0: 'w.txt',
        1: 'z.txt',
        2: 'r.txt',
        3: 'tde.txt',
    }

action_dict = {
        0: 'forward',
        1: 'backward',
        2: 'go left',
        3: 'go right',
        4: 'idle',
    }

# return true if increasing, false if decreasing or no change
#@jit(nopython=True)
def get_roc(_data, _record, _type):
    delta = 0.0
    for i in range(_record.size):
        delta += (_data - _record[i])
    delta /= float(_record.size)
    if _type == 0:  # photo data
        delta = (delta + 511.50) / 1023.0
    else:  # temp data
        delta = (delta + 10.0) / 20.0
    return delta > 0.5

def get_reward(_temp, _collision):
    r = 5.0 if (_temp >= 23.5 and _temp <= 25.5) else -1.0
    r = r - 5.0 if _collision else r
    return r

#@jit(nopython=True)
def get_echo_index(_data):
    return math.floor(((_data - 3.0) / 90.0) * 4)

def update_record(_record, _data):
    _record = np.roll(_record, -1)
    _record[-1:] = _data
    return _record

def byte_data_to_list():
    byte_data = serial.readline()
    data = byte_data[0:len(byte_data) - 2].decode("utf-8")
    data_list = np.array(data.split(), dtype=float)
    return data_list

def store_t(_t):
    f_path = 't.txt'
    with open(f_path, "w") as f:
        f.write(str(_t))

def read_t():
    f_path = 't.txt'
    with open(f_path, "r") as f:
        stored_t = int(f.readline())
    return stored_t

def write_to_record(_mode, _avg_val):
    f_path = file_path_dict.get(_mode, 'r.txt')
    with open(f_path, "a") as record:
        record.write("\n{}".format(_avg_val))
    record.close()

#@jit(nopython=True)
def write_to_memory(_mode, _ndarray):
    f_path = file_path_dict.get(_mode, 'w.txt')
    with open(f_path, "w+") as memory:
        memory.truncate(0)
        row = ''
        for i in range(4):
            ndarr_i = np.ndarray.copy(_ndarray[i])
            arr_i = np.ndarray.flatten(ndarr_i)
            for j in range(arr_i.size):
                row += (str(arr_i[j]) + ' ')
            row += '\n'
        memory.write(row)
    memory.close()

#@jit(nopython=True)
def read_from_memory(_mode, _ndarray):
    f_path = file_path_dict.get(_mode, 'w.txt')
    with open(f_path, "r") as memory:
        data_list = np.array([])
        for i in range(4):
                data_row = np.array(memory.readline().split(), dtype=float)
                data_list = np.append(data_list, data_row)
        updated_ndarray = np.reshape(data_list, _ndarray.shape)
        return updated_ndarray

#@jit(nopython=True)
def get_bumper_state(_l, _r):
    if _l or _r:
        return 1
    return 0

#@jit(nopython=True)
def bound_temp_data(_data):
    _data = 29.9 if _data >= 29.9 else _data
    _data = 20.0 if _data < 20.0 else _data
    return _data

#@jit(nopython=True)
def bound_photo_data(_data):
    _data = 511.0 if _data >= 511.0 else _data
    _data = 0.0 if _data < 0.0 else _data
    return _data

#@jit(nopython=True)
def bound_uss_data(_data):
    _data = 3.0 if _data >= 200.0 else _data
    _data = 91.5 if _data >= 91.5 else _data
    _data = 3.0 if _data < 3.0 else _data
    return _data

#@jit(nopython=True)
def get_ap(_print_mode):
    rand = random.random()
    if rand > eps:
        q_x = np.zeros(action_num, dtype=np.float32)
        for n in range(tile_num):
            for m in range(action_num):
                q_x[m] += w[n, gen_x_i[n, 0], gen_x_i[n, 1], gen_x_i[n, 2], gen_x_i[n, 3],
                            gen_x_i[n, 4], gen_x_i[n, 5], gen_x_i[n, 6], m]

        # if _print_mode: print(f'action-values:      {q_x}')

        qp = np.amax(q_x)
        ap_arr = np.where(q_x == qp)[0]
        ret_ap = ap_arr[np.random.randint(ap_arr.size)]
        if _print_mode: print(f'exploitative action:    {ap}')
    else:
        ret_ap = np.random.randint(action_num)
        if _print_mode: print(f'explorative action:  {ap}')
    return ret_ap

#@jit(nopython=True)
def generalize(_gen_x_i):

    for n in range(tile_num):

        rand = np.random.randint(-1, 2)
        _gen_x_i[n, 0] = math.floor(bound_photo_data(avg_photo + (photo_unit * rand)) / 511.50 * state_agg)
        _gen_x_i[n, 1] = x[1]
        rand = np.random.randint(-1, 2)
        _gen_x_i[n, 2] = math.floor(((bound_temp_data(temp + (temp_unit * rand)) - 20.0) / 10.0) * state_agg)
        _gen_x_i[n, 3] = x[3]
        rand = np.random.randint(-1, 2)
        _gen_x_i[n, 4] = get_echo_index(bound_uss_data(uss_l + (uss_unit * rand)))
        rand = np.random.randint(-1, 2)
        _gen_x_i[n, 5] = get_echo_index(bound_uss_data(uss_r + (uss_unit * rand)))
        _gen_x_i[n, 6] = x[6]

#@jit(nopython=True)
def get_partial_tde_0(_a, _r, _z):
    partial_tde = _r
    for i in range(tile_num):
        partial_tde -= w[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2],
                         gen_x_i[i, 3], gen_x_i[i, 4], gen_x_i[i, 5], gen_x_i[i, 6], _a]

        _z[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2], gen_x_i[i, 3],
                                    gen_x_i[i, 4], gen_x_i[i, 5], gen_x_i[i, 6], _a] += 1
    return partial_tde

#@jit(nopython=True)
def get_partial_tde_1(_ap, _partial_tde):
    for i in range(tile_num):
        _partial_tde += gamma * w[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2],
                                  gen_x_i[i, 3], gen_x_i[i, 4], gen_x_i[i, 5], gen_x_i[i, 6], _ap]
    return _partial_tde

#@jit(nopython=True)

def update_w_z(_w, _z):

    _w += alpha * tde * _z

    decay = (gamma * lambda_decay)
    _z[np.where(_z != 0)] *= decay


if __name__ == '__main__':

    feature_num = 7
    action_num = 5
    tile_num = 4

    state_agg = 6

    print_mode = True

    # w = [tilings][avg photo][photo roc][temp][temp roc][sonic left][sonic right][bumper][action-num]
    #         4         6          2       6        2          4           4          2        5
    w = np.zeros((tile_num, state_agg, 2, state_agg, 2, 4, 4, 2, action_num), dtype=np.float32)
    # or load from memory
    # w = read_from_memory(True, w)

    z = np.zeros((tile_num, state_agg, 2, state_agg, 2, 4, 4, 2, action_num), dtype=np.float32)
    # or load from memory
    # z = read_from_memory(False, z)

    # t = 0
    t = read_t()

    eps = 0.01
    lambda_decay = 0.75
    alpha = 0.001
    gamma = 0.97

    # units per metric: width / tile_num
    photo_unit = 12.7875        # range of 511.50, from 0 - 511.50
    temp_unit = 0.5             # range of 20, from 23 - 43 C
    uss_unit = 5.0              # range of 60, from 2 - 62 cm

    x = np.zeros(feature_num, dtype=int)
    xp = np.zeros(feature_num, dtype=int)
    gen_x_i = np.zeros((tile_num, feature_num), dtype=int)
    gen_xp_i = np.zeros((tile_num, feature_num), dtype=int)

    photo_record = np.zeros(3, dtype=np.float32)
    temp_record = np.zeros(3, dtype=np.float32)

    a, ap = 0, 0
    tde = 0.0

    avg_r = 0.0
    avg_tde = 0.0

    # initialize data records
    for i in range(3):

        # no movement and no vector updates yet.
        # take a random action to init.
        a = np.random.randint(action_num)
        transmit_action = str(a) + '\n'
        serial.write(transmit_action.encode('utf-8'))
        serial.flush()
        serial.reset_input_buffer()

        if print_mode: print(f"Take action -> {action_dict.get(a, '???')}\n\n")

        while serial.in_waiting <= 5:
            continue

        data_list = byte_data_to_list()

        if print_mode:
            print(f'--------------------------------------------------------\n\nt = {t}')
            print(data_list)

        l_photo = data_list[0]
        r_photo = data_list[1]
        l_photo = bound_photo_data(l_photo)
        r_photo = bound_photo_data(r_photo)
        avg_photo = (l_photo + r_photo) / 2.0

        temp = data_list[2]
        temp = bound_temp_data(temp)

        l_bump = int(data_list[6])
        r_bump = int(data_list[7])

        if i == 2:

            uss_r = bound_uss_data(data_list[3] + (0.5 * data_list[5]))
            uss_l = bound_uss_data(data_list[4] + (0.5 * data_list[5]))

            x[0] = math.floor((avg_photo / 511.5) * state_agg)
            x[1] = get_roc(avg_photo, photo_record, 0)
            x[2] = math.floor(((temp - 20.0) / 10.0) * state_agg)
            x[3] = get_roc(temp, temp_record, 1)
            x[4] = get_echo_index(uss_l)
            x[5] = get_echo_index(uss_r)
            x[6] = get_bumper_state(l_bump, r_bump)

            if print_mode:
                print(f'received data:  {data_list}\n')

                print(f'light:          {avg_photo}')
                print(f'temp:           {temp}')
                print(f'obstacles:      {data_list[3]} cm at 30, {data_list[5]} cm at 90, {data_list[4]} cm at 135')
                print(f'collision:      {(l_bump | r_bump)}\n')

                print(f'feature:        {x}')

            # populate new gen_x_i by shifting
            generalize(gen_x_i)

        update_record(photo_record, avg_photo)
        update_record(temp_record, temp)

    while True:

        if print_mode: print("Take action -> {action_dict.get(a, '???')}\n\n")

        instruction = str(a) + '\n'
        serial.write(instruction.encode('utf-8'))
        serial.flush()
        serial.reset_input_buffer()

        # wait for microcontroller
        while serial.in_waiting < 5:
            continue

        # observe S
        data_list = byte_data_to_list()

        if print_mode:
            print(f'--------------------------------------------------------\n\nt = {t}')
            print(data_list)

        l_photo = data_list[0]
        r_photo = data_list[1]
        l_photo = bound_photo_data(l_photo)
        r_photo = bound_photo_data(r_photo)
        avg_photo = (l_photo + r_photo) / 2.0

        temp = data_list[2]
        temp = bound_temp_data(temp)

        uss_r = bound_uss_data(data_list[3] + (0.5 * data_list[5]))
        uss_l = bound_uss_data(data_list[4] + (0.5 * data_list[5]))

        l_bump = int(data_list[6])
        r_bump = int(data_list[7])

        x[0] = math.floor((avg_photo / 511.50) * state_agg)
        x[1] = get_roc(avg_photo, photo_record, 0)
        x[2] = math.floor(((temp - 20.0) / 10.0) * state_agg)
        x[3] = get_roc(temp, temp_record, 1)
        x[4] = get_echo_index(uss_l)
        x[5] = get_echo_index(uss_r)
        x[6] = get_bumper_state(l_bump, r_bump)

        if print_mode:
            print(f'received data:  {data_list}\n')

            print(f'light:          {avg_photo}')
            print(f'temp:           {temp}')
            print(f'obstacles:      {data_list[3]} cm at 30, {data_list[5]} cm at 90, {data_list[4]} cm at 135')
            print(f'collision:      {(l_bump | r_bump)}\n')

            print(f'feature:        {x}')

        # collect reward
        r = get_reward(temp, bool(x[6]))
        avg_r += r

        # generalize and calculate partial td error
        partial_td_err = get_partial_tde_0(a, r, z)

        # populate new gen_x_i by shifting
        generalize(gen_x_i)

        # find action with the highest value for these features
        ap = get_ap(print_mode)
        tde = get_partial_tde_1(ap, partial_td_err)

        if print_mode:
            print(f'\n>> TD error   >>  {tde}')
            print(f'>> Reward   >>  {r}\n')

        # update weight vector for last time step
        update_w_z(w, z)

        avg_tde += tde

        # document average TD errors and reward per 10 steps
        if t % 10 == 0:
            write_to_record(2, (avg_r / 10.0))
            write_to_record(3, (avg_tde / 10.0))
            avg_r, avg_tde = 0.0, 0.0

        # write weight and trace vector into memory
        if t % 500 == 0 and t != 0:
            write_to_memory(0, w)
            write_to_memory(1, z)
            store_t(t)
            print(f't: {t}')
            print(
                'memory stored ---------------------------------------------------------------------------------\n\n')

        photo_record = update_record(photo_record, avg_photo)
        temp_record = update_record(temp_record, temp)

        a = ap
        t += 1


