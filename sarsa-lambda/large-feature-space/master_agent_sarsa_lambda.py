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
    r = 0.0 if (_temp >= 25.5 and _temp <= 27.5) else -1.0
    r = r - 1.0 if _collision else r
    return r

def get_echo_index(_data):
    return math.floor(((_data - 2.0) / 60.0) * 5.0)

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
        f.writelines(_t)

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

def read_from_memory(_mode, _ndarray):
    f_path = file_path_dict.get(_mode, 'w.txt')
    with open(f_path, "r") as memory:
        data_list = np.array([])
        for i in range(4):
                data_row = np.array(memory.readline().split(), dtype=float)
                data_list = np.append(data_list, data_row)
        updated_ndarray = np.reshape(data_list, _ndarray.shape)
        return updated_ndarray

def get_bumper_state(_l, _r):
    if _l or _r:
        return 1
    return 0

def bound_temp_data(_data):
    _data = 29.9 if _data >= 29.9 else _data
    _data = 20.0 if _data < 20.0 else _data
    return _data

def bound_photo_data(_data):
    _data = 511.0 if _data >= 511.0 else _data
    _data = 0.0 if _data < 0.0 else _data
    return _data

def bound_uss_data(_data):
    _data = 61.5 if _data >= 61.5 else _data
    _data = 2.0 if _data < 2.0 else _data
    return _data

def generalize(_gen_x_i, _i, _l_photo, _r_photo, _temp, _uss30, _uss90, _uss135, _x):

    rand = np.random.randint(-1, 2)
    _gen_x_i[i, 0] = math.floor(bound_photo_data(_l_photo + (photo_unit * rand)) / 511.50 * 15.0)

    rand = np.random.randint(-1, 2)
    _gen_x_i[i, 1] = math.floor(bound_photo_data(_r_photo + (photo_unit * rand)) / 511.50 * 15.0)

    _gen_x_i[i, 2] = _x[2]

    rand = np.random.randint(-1, 2)
    _gen_x_i[i, 3] = math.floor(((bound_temp_data(_temp + (temp_unit * rand)) - 20.0) / 10.0) * 15.0)

    _gen_x_i[i, 4] = _x[4]

    rand = np.random.randint(-1, 2)
    _gen_x_i[i, 5] = get_echo_index(bound_uss_data(_uss30 + (uss_unit * rand)))

    rand = np.random.randint(-1, 2)
    _gen_x_i[i, 6] = get_echo_index(bound_uss_data(_uss90 + (uss_unit * rand)))

    rand = np.random.randint(-1, 2)
    _gen_x_i[i, 7] = get_echo_index(bound_uss_data(_uss135 + (uss_unit * rand)))

    _gen_x_i[i, 8] = _x[8]

    return _gen_x_i


if __name__ == '__main__':

    feature_num = 9
    action_num = 5
    tile_num = 16

    print_mode = True

    # w = [tilings][left photo][right photo][photo roc][temp][temp roc][sonic 90][sonic 45][sonic 135][bumper][action-num]
    #         4         15           15          2       15       2        5         5         5          4        5
    w = np.zeros((tile_num, 15, 15, 2, 15, 2, 5, 5, 5, 2, action_num))
    # or load from memory
    # w = read_from_memory(True, w)

    z = np.zeros((tile_num, 15, 15, 2, 15, 2, 5, 5, 5, 2, action_num))
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

    photo_record = np.zeros(3)
    temp_record = np.zeros(3)

    a, ap = 0, 0
    qp = 0.0
    td_err = 0.0

    avg_r = 0.0
    avg_td_err = 0.0

    # initialize data records
    for i in range(3):

        # no movement and no vector updates yet.
        # take a random action to init.
        a = np.random.randint(action_num)
        transmit_action = str(a) + '\n'
        serial.write(transmit_action.encode('utf-8'))
        serial.flush()
        serial.reset_input_buffer()

        while serial.in_waiting <= 5:
            continue

        data_list = byte_data_to_list()
        
        if print_mode:
            print('--------------------------------------------------------\n\nt = {}'.format(t))
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

            uss30 = bound_uss_data(data_list[3])
            uss135 = bound_uss_data(data_list[4])
            uss90 = bound_uss_data(data_list[5])

            x[0] = math.floor((l_photo / 511.50) * 15.0)
            x[1] = math.floor((r_photo / 511.50) * 15.0)
            x[2] = get_roc(avg_photo, photo_record, 0)
            x[3] = math.floor(((temp - 20.0) / 10.0) * 15.0)
            x[4] = get_roc(temp, temp_record, 1)
            x[5] = get_echo_index(uss30)
            x[6] = get_echo_index(uss90)
            x[7] = get_echo_index(uss135)
            x[8] = get_bumper_state(l_bump, r_bump)
            
            if print_mode:

                print('received data:   {}\n'.format(data_list))

                print('light:           {}'.format(avg_photo))
                print('temp:            {}'.format(temp))
                print('obstacles:       {} cm at 30, {} cm at 90, {} cm at 135'.format(uss30, uss90, uss135))
                print('collision:       {}\n'.format((l_bump | r_bump)))

                print('feature:         {}'.format(x))

            for i in range(tile_num):
                gen_x_i = generalize(gen_x_i, i, r_photo, r_photo, temp, uss30, uss90, uss135, x)

        update_record(photo_record, avg_photo)
        update_record(temp_record, temp)

    a = np.random.randint(action_num)
    while True:

            if print_mode: print("Take action -> {}\n\n".format(action_dict.get(a, '???')))

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
                print('--------------------------------------------------------\n\nt = {}'.format(t))
                print(data_list)

            l_photo = data_list[0]
            r_photo = data_list[1]
            l_photo = bound_photo_data(l_photo)
            r_photo = bound_photo_data(r_photo)
            avg_photo = (l_photo + r_photo) / 2.0

            temp = data_list[2]
            temp = bound_temp_data(temp)

            uss30 = bound_uss_data(data_list[3])
            uss135 = bound_uss_data(data_list[4])
            uss90 = bound_uss_data(data_list[5])

            l_bump = int(data_list[6])
            r_bump = int(data_list[7])

            x[0] = math.floor((l_photo / 511.50) * 15.0)
            x[1] = math.floor((r_photo / 511.50) * 15.0)
            x[2] = get_roc(avg_photo, photo_record, 0)
            x[3] = math.floor(((temp - 20.0) / 10.0) * 15.0)
            x[4] = get_roc(temp, temp_record, 1)
            x[5] = get_echo_index(uss30)
            x[6] = get_echo_index(uss90)
            x[7] = get_echo_index(uss135)
            x[8] = get_bumper_state(l_bump, r_bump)

            if print_mode:

                print('received data:   {}\n'.format(data_list))

                print('light:           {}'.format(avg_photo))
                print('temp:            {}'.format(temp))
                print('obstacles:       {} cm at 30, {} cm at 90, {} cm at 135'.format(uss30, uss90, uss135))
                print('collision:       {}\n'.format((l_bump | r_bump)))

                print('feature:         {}'.format(x))

            # collect reward
            r = get_reward(temp, bool(x[8]))
            avg_r += r

            td_err = r

            # generalize and calculate partial td error
            for i in range(tile_num):

                td_err -= w[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2], gen_x_i[i, 3],
                            gen_x_i[i, 4], gen_x_i[i, 5], gen_x_i[i, 6], gen_x_i[i, 7], gen_x_i[i, 8], a]

                z[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2], gen_x_i[i, 3],
                  gen_x_i[i, 4], gen_x_i[i, 5], gen_x_i[i, 6], gen_x_i[i, 7], gen_x_i[i, 8], a] += 1

                gen_x_i = generalize(gen_x_i, i, r_photo, r_photo, temp, uss30, uss90, uss135, x)


            # find action with the highest value for these features
            rand = random.random()
            if rand > eps:
                q_x = np.zeros(action_num)
                for i in range(tile_num):
                    for j in range(action_num):
                        q_x[j] += w[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2], gen_x_i[i, 3],
                                     gen_x_i[i, 4], gen_x_i[i, 5], gen_x_i[i, 6], gen_x_i[i, 7], gen_x_i[i, 8], j]

                if print_mode: print('action-values:       {}'.format(np.around(q_x, decimals=2)))

                qp = np.amax(q_x)
                ap_arr = np.where(q_x == qp)[0]
                ap = ap_arr[np.random.randint(ap_arr.size)]
                if print_mode: print('exploitative action: {}'.format(action_dict.get(ap)))
            else:
                ap = np.random.randint(action_num)
                if print_mode: print('explorative action:  {}'.format(action_dict.get(ap)))

            for i in range(tile_num):
                td_err += gamma * w[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2], gen_x_i[i, 3],
                                    gen_x_i[i, 4], gen_x_i[i, 5], gen_x_i[i, 6], gen_x_i[i, 7], gen_x_i[i, 8], ap]

            if print_mode:
                print('\n>> TD error >> {}'.format(td_err))
                print('>> Reward   >> {}\n'.format(r))

            # update weight vector for last time step
            w += alpha * td_err * z
            z = gamma * lambda_decay * z

            avg_td_err += td_err

            # document average TD errors and reward per 10 steps
            if t % 10 == 0:
                write_to_record(2, (avg_r / 10.0))
                write_to_record(3, (avg_td_err / 10.0))
                avg_r, avg_td_err = 0.0, 0.0

            # write weight and trace vector into memory
            if t % 500 == 0:
                write_to_memory(0, w)
                write_to_memory(1, z)
                store_t(t)
                print('t: {}'.format(t))
                print(
                    'memory stored ---------------------------------------------------------------------------------\n\n')

            photo_record = update_record(photo_record, avg_photo)
            temp_record = update_record(temp_record, temp)

            q = qp
            a = ap
            t += 1
