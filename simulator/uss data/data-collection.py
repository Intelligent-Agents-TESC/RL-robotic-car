import sys
import serial

# serially connected arduino
serial = serial.Serial('/dev/ttyACM0', 115200, timeout=0.5)

def collect_data(_deg, _dist):

    f_name = "deg_{0}_at_{1}_cm.txt".format(str(_deg), str(_dist))

    with open(f_name, "w") as f:

        for t in range(100):

            # wait for arduino to send data
            while serial.in_waiting < 1:
                continue

            # convert bytes to string
            byte_data = serial.readline()
            data = byte_data[0:len(byte_data) - 2].decode("utf-8")
            data += '\n'
            print(data)
            f.write(data)

            transmit_action = str(4) + '\n'
            serial.write(transmit_action.encode('utf-8'))
            serial.flush()
            serial.reset_input_buffer()

    f.close()

if __name__ == '__main__':
    deg = sys.argv[1]
    dist = sys.argv[2]
    collect_data(deg, dist)
