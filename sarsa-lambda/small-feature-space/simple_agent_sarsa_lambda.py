import matplotlib.pyplot as plt
import numpy as np
import serial
import random
import math

# servant arduino
serial = serial.Serial('/dev/ttyACM0', 115200, timeout=None)

file_path_dict = {
        0: 'w.txt',
        1: 'z.txt',
        2: 'r.txt',
        3: 'tde.txt',
        4: 'arb_tde.txt'
    }

def byte_data_to_list():
    byte_data = serial.readline()
    data = byte_data[0:len(byte_data) - 2].decode("utf-8")
    data_list = np.array(data.split(), dtype=float)
    return data_list

def get_reward(_temp, _collision):
    r = 0.0 if (_temp >= 25.5 and _temp <= 27.5) else -1.0
    r = r - 1.0 if _collision else r
    return r

def get_echo_i(_data):
    i = math.floor(((_data - 2.0) / 60.0) * 20.0)
    i = 19 if i > 19 else i
    return i

def get_photo_i(_data):
    i = math.floor((_data / 511.50) * 20.0)
    i = 19 if i > 19 else i
    return i

def get_temp_i(_data):
    i = math.floor(((_data - 23.0) / 20.0) * 20.0)
    i = 19 if i > 19 else i
    return i

def rd_t():
    f_path = 't.txt'
    with open(f_path, "r") as f:
        stored_t = int(f.readline())
    return stored_t

def store_t(_t):
    f_path = 't.txt'
    with open(f_path, "w") as f:
        f.writelines(_t)

def wr_record(_mode, _avg_val):
    f_path = file_path_dict.get(_mode, 'r.txt')
    with open(f_path, "a") as record:
        record.write("\n{}".format(_avg_val))
    record.close()

def wr_memory(_mode, _ndarray):
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

def rd_memory(_mode, _ndarray):
    f_path = file_path_dict.get(_mode, 'w.txt')
    with open(f_path, "r") as memory:
        data_list = np.array([])
        for i in range(4):
                data_row = np.array(memory.readline().split(), dtype=float)
                data_list = np.append(data_list, data_row)
        updated_ndarray = np.reshape(data_list, _ndarray.shape)
        return updated_ndarray

def sim_temp(_current_temp, _photo_data):

    # 0 -> -0.125, 100 -> -0.016
    delta = ((math.log((_photo_data + 0.2) / 0.1)) / 5.5) - 0.39
    temp = _current_temp + delta

    # bound temp data: range of 20, from 23 - 43 C
    temp = bound_temp(temp)

    return temp

# range of 511.50, from 0 - 511.50
def bound_photo(_data):
    _data = 511.0 if _data > 511.0 else _data
    _data = 0.0 if _data < 0.0 else _data
    return _data

def bound_temp(_data):
    _data = 43.0 if _data >= 43.0 else _data
    _data = 23.0 if _data < 23.0 else _data
    return _data

# range of 60, from 2 - 62 cm
def bound_uss(_data):
    _data = 62.0 if _data > 62.0 else _data
    _data = 2.0 if _data < 2.0 else _data
    return _data


def live_plotter(x_tde, y_tde, line_tde, y_r, line_r, identifier=''):

    if line_tde == [] and line_r == []:

        # call to matplotlib for dynamic plotting
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        # turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        # turn off x ticks for the upper subplot
        ax1.tick_params(axis='x', labelcolor='w', top=False, bottom=False, left=False, right=False)

        # create a variable for the line so we can later update it
        line_tde, = ax1.plot(x_tde, y_tde, '-o', color='blue', alpha=0.8, linewidth=1)
        line_r = ax2.plot(x_tde, y_r, '-o', color='red', alpha=0.8, linewidth=1)

        # update plot label/title
        plt.ylabel('TD Error')
        plt.title('{}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line_tde.set_ydata(y_tde)
    line_r.set_ydata(y_r)

    # adjust limits if new data goes beyond bounds
    if np.min(y_tde) <= line_tde.axes.get_ylim()[0] or np.max(y_tde) >= line_tde.axes.get_ylim()[1]:
        plt.ylim([np.min(y_tde) - np.std(y_tde), np.max(y_tde) + np.std(y_tde)])

    # this pauses the data so the figure/axis can catch up
    plt.pause(0.1)

    # return line so we can update it again in the next iteration
    return line_tde, line_r

if __name__ == '__main__':

    feature_num = 4
    action_num = 4
    tile_num = 8

    # w = [tilings][avg. photo][temp][uss][bumper][action-num]
    #         8         20       20    20     2        3
    w = np.zeros((tile_num, 20, 20, 20, 2, action_num))
    # or load from memory
    # w = read_from_memory(True, w)

    z = np.zeros((tile_num, 20, 20, 20, 2, action_num))
    # or load from memory
    # z = read_from_memory(False, z)

    # t = 0
    t = rd_t()

    eps = 0.01
    lambda_decay = 0.8
    alpha = 0.025
    gamma = 0.9

    # units per metric: width / tile_num
    photo_unit = 2.575     # range of 511.50, from 0 - 511.50
    temp_unit = 0.25      # range of 20, from 23 - 43 C
    uss_unit = 0.5       # range of 60, from 2 - 62 cm

    x = np.zeros(feature_num, dtype=int)
    xp = np.zeros(feature_num, dtype=int)
    gen_x_i = np.zeros((tile_num, feature_num), dtype=int)
    gen_xp_i = np.zeros((tile_num, feature_num), dtype=int)

    a, ap = 0, 0
    q, qp = 0.0, 0.0
    td_err = 0.0
    r = 0.0

    avg_r = 0.0
    avg_td_err = 0.0

    temp = 23.5

    # initialize data records
    for i in range(3):

        while serial.in_waiting <= 5:
            continue

        #     0        1      2       3         4        5
        # [l_photo][r_photo][temp][watch90][l_bumper][r_bumper]
        data_list = byte_data_to_list()

        l_photo = bound_photo(data_list[0])
        r_photo = bound_photo(data_list[1])
        avg_photo = (l_photo + r_photo) / 2.0

        temp = sim_temp(temp, avg_photo)

        watch90 = data_list[2]
        uss = bound_uss(watch90)

        l_bump = int(data_list[3])
        r_bump = int(data_list[4])

        if i == 2:

            x[0] = get_photo_i(avg_photo)
            x[1] = get_temp_i(temp)
            x[2] = get_echo_i(uss)
            x[3] = int(l_bump | r_bump)

        # no movement and no vector updates during init.
        instruction = str(6) + '\n'
        serial.write(instruction.encode('utf-8'))
        serial.flush()
        serial.reset_input_buffer()

    while True:

        if serial.in_waiting > 5:

            data_list = byte_data_to_list()
            print(data_list)

            l_photo = bound_photo(data_list[0])
            r_photo = bound_photo(data_list[1])
            avg_photo = (l_photo + r_photo) / 2.0

            temp = sim_temp(temp, avg_photo)

            watch90 = data_list[3]
            uss = bound_uss(watch90)

            l_bump = int(data_list[4])
            r_bump = int(data_list[5])

            x[0] = get_photo_i(avg_photo)
            x[1] = get_temp_i(temp)
            x[2] = get_echo_i(uss)
            x[3] = int(l_bump | r_bump)
            print('xp: {}'.format(xp))

                        # collect reward
            r = get_reward(temp, bool(xp[3]))
            avg_r += r
            td_err = r

            # generalize
            for i in range(tile_num):
                rand = np.random.randint(-1, 2)
                gen_x_i[i, 0] = get_photo_i(bound_photo(avg_photo + (photo_unit * rand)))
                rand = np.random.randint(-1, 2)
                gen_x_i[i, 1] = get_temp_i(bound_temp(temp + (temp_unit * rand)))
                rand = np.random.randint(-1, 2)
                gen_x_i[i, 2] = get_echo_i(bound_uss(uss + (uss_unit * rand)))
                gen_x_i[i, 3] = xp[3]
                
                td_err -= w[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2], gen_x_i[i, 3], a]
                z[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2], gen_x_i[i, 3], a] += 1


            # find action with the highest value for these features
            rand = random.random()
            if rand > eps:
                q_x = np.zeros(action_num)
                for i in range(tile_num):
                    for j in range(action_num):
                        q_x[j] += w[i, gen_xp_i[i, 0], gen_xp_i[i, 1], gen_xp_i[i, 2], gen_xp_i[i, 3], j]
                qp = np.amax(q_x)
                ap_arr = np.where(q_x == qp)[0]
                ap = ap_arr[np.random.randint(ap_arr.size)]
            else:
                ap = np.random.randint(action_num)

            print("action: ")
            print(ap)

            instruction = str(ap) + '\n'
            serial.write(instruction.encode('utf-8'))
            serial.flush()
            serial.reset_input_buffer()

            for i in range(tile_num):
                td_err += gamma * w[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2], gen_x_i[i, 3], ap]

            # update weight vector for last time step
            w += alpha * td_err * z
            z = gamma * lambda_decay * z

            avg_td_err += td_err

            # document average TD errors and reward per 10 steps
            if t % 10 == 0 and t != 0:
                wr_record(2, (avg_r / 10.0))
                wr_record(3, (avg_td_err / 10.0))
                avg_r, avg_td_err = 0.0, 0.0

            # write weight and trace vector into memory
            if t % 500 == 0 and t != 0:
                wr_memory(0, w)
                wr_memory(1, z)
                store_t(t)
                print('t: {}'.format(t))
                print(
                    '\n\n\nmemory stored ---------------------------------------------------------------------------------\n\n\n')
                    
            a = ap
            t += 1


