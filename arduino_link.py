import time
import serial   # from the pyserial library:  pip install pyserial 

# ---------- CHANGE THIS to your Arduino's port ----------
# Windows: 'COM3' (find it in Arduino IDE -> Tools -> Port)
# Raspberry Pi / Linux: '/dev/ttyACM0'      Mac: '/dev/cu.usbmodem...'
SERIAL_PORT = 'COM6'
BAUD_RATE = 9600
# --------------------------------------------------------


def connect(port=SERIAL_PORT):
    #Open the connection to the Arduino.
    ser = serial.serial_for_url(port, baudrate=BAUD_RATE, timeout=1)
    time.sleep(2)              # the Arduino restarts when the port opens - wait for it
    ser.reset_input_buffer()   # throw away the startup "READY" message
    return ser


def send_command(ser, target, action):
    #Send one command like ('ALL', 'HIGH') or ('Z2', 'LOW'). Returns Arduino's reply."""
    line = f"{target}:{action}\n"
    ser.write(line.encode())
    reply = ser.readline().decode().strip()
    return reply


def hardware_test():
    """Run this file directly to check that the Arduino, fans and LEDs respond."""
    ser = connect()
    print("Connected. Watch your fans and LEDs.\n")

    steps = [
        ('ALL', 'OFF'), ('ALL', 'LOW'), ('ALL', 'MEDIUM'), ('ALL', 'HIGH'), ('ALL', 'OFF'),
        ('Z1', 'HIGH'), ('Z2', 'HIGH'), ('Z3', 'HIGH'), ('Z4', 'HIGH'), ('ALL', 'OFF'),
    ]
    for target, action in steps:
        reply = send_command(ser, target, action)
        print(f"Sent {target}:{action:<7} -> Arduino says: {reply}")
        time.sleep(2)

    ser.close()
    print("\nTest finished.")


if __name__ == '__main__':
    hardware_test()
