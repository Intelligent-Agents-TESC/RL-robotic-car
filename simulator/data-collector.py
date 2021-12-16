import sys
import serial
import numpy as np

# serially connected arduino
serial = serial.Serial('/dev/ttyACM0', 115200, timeout=0.5)

def collect_data(_y, _x):

    f_name = "{0}_{1}.txt".format(str(_y), str(_x))

    with open(f_name, "w") as f:

        for t in range(80):

            # wait for arduino to send data
            while serial.in_waiting < 5:
                continue

            # convert bytes to string
            byte_data = serial.readline()
            data = byte_data[0:len(byte_data) - 2].decode("utf-8")
            data += '\n'
            f.write(data)

            transmit_action = str(4) + '\n'
            serial.write(transmit_action.encode('utf-8'))
            serial.flush()
            serial.reset_input_buffer()



    f.close()

if __name__ == '__main__':
    y = sys.argv[1]
    x = sys.argv[2]
    collect_data(y, x)

