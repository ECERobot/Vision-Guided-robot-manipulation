import serial
import time
import numpy as np

class DensoRobotControl:
    # DENSO ROBOT ALWAYS SENDS AND EXPECTS A CR AT THE END OF EACH MESSAGE ('\r' character or number 13)
    
    def __init__(self, port=None, baud_rate=19200):
        self.msg_timeout = 0.1  # Message timeout in seconds
        self.pos_timeout = 1500  # Position timeout in seconds
        self.serial_com = None

        if port is not None:
            self.connect(port, baud_rate)

    def connect(self, port, baud_rate=19200):
        try:
            self.serial_com = serial.Serial(
                port=port,
                baudrate=baud_rate,
                timeout=self.msg_timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            time.sleep(0.5)  # Delay to allow connection to stabilize
            print("Connected!")
        except serial.SerialException as e:
            raise RuntimeError(f"Connection failed: {str(e)}") from e

    def isConnected(self):
        return self.serial_com and self.serial_com.is_open

    def setTimeout(self, msg_time_ms=100, pos_time_ms=1500000):
        self.msg_timeout = msg_time_ms / 1000  # Convert ms to seconds
        self.pos_timeout = pos_time_ms / 1000  # Convert ms to seconds
        if self.serial_com:
            self.serial_com.timeout = self.msg_timeout

    def getJoint(self):
        if not self.isConnected():
            raise RuntimeError("Not connected to the robot")

        # Send command to get joint angles
        command = "2\r"
        try:
            self.serial_com.write(command.encode('utf-8'))
        except serial.SerialTimeoutException:
            print("Write timeout occurred")
            return -2

        # Read response
        try:
            response = self.serial_com.read_until(b'\r').decode('utf-8').strip()
        except serial.SerialException:
            print("Read error occurred")
            return -1

        # Parse joint data
        try:
            joint_data = list(map(float, response.split()))
            if len(joint_data) != 6:
                print("Error: The message must return 6 joint values")
                return -3
            print(f"Joint data: {joint_data}")
            return joint_data
        except ValueError:
            print("Error converting data to float")
            return -2

    def getPosition(self, verbose = False):
        if not self.isConnected():
            raise RuntimeError("Not connected to the robot")

        # Define the command
        command = "3\r"  # GETJOINT command type

        # Send the command
        try:
            self.serial_com.write(command.encode('utf-8'))
        except serial.SerialTimeoutException:
            print("Error: Write timeout occurred")
            return -2

        # Receive joint data
        try:
            receive_msg = self.serial_com.read_until(b'\r').decode('utf-8').strip()
        except serial.SerialTimeoutException:
            print(" MESSAGE: NOT RECEIVED. TIME OUT.")
            print(" INFO   : Please, check that the robot is ready to work.")
            return -1
        except serial.SerialException:
            print(" MESSAGE: NOT RECEIVED. COMMUNICATION ERROR.")
            print(" INFO   : Check the RS-232 connection.")
            return -4

        # Parse the received message
        try:
            tokens = receive_msg.split()
            joint_state = [float(token) for token in tokens if token.strip()]
        except ValueError:
            print(" MESSAGE: WRONG FORMAT RECEIVED.")
            print(" INFO   : Error converting the data into double type.")
            return -2

        if len(joint_state) != 7:
            print(" MESSAGE: WRONG FORMAT RECEIVED.")
            print(" INFO   : The message must return 7 position data (6 pos + Fig).")
            return -3

        # Assign values
        x, y, z, rx, ry, rz = joint_state[:6]
        fig = int(joint_state[6])

        # Log results
        if  verbose:
            print("\n====== POSITION STATE =========================================")
            print(f" X  : {x}")
            print(f" Y  : {y}")
            print(f" Z  : {z}")
            print(f" RX : {rx}")
            print(f" RY : {ry}")
            print(f" RZ : {rz}")
            print(f" Fig: {fig}")
            print("--------------------------------------------------------------")

        return np.array([x, y, z, rx, ry, rz]), fig


    def moveJoint(self, joint, speed=15, gripper=False, figure=5,verbose = False):
        # Define the command
        command = f"0\r{joint[0]},{joint[1]},{joint[2]},{joint[3]},{joint[4]},{joint[5]},{speed},{int(gripper)},{figure}\r"

        # Log the command details
        if verbose:
            print("\n====== Go To ====================================================")
            print(f" ANGLE  : {joint[0]}, {joint[1]}, {joint[2]}, {joint[3]}, {joint[4]}, {joint[5]}")
            print(f" SPEED  : {speed}")
            print(f" FIG    : {figure}")
            print(f" GRIPPER: {'CLOSE' if gripper else 'OPEN'}")
            print("-------------------------------------------------------------------")

        # Send the command
        try:
            self.serial_com.write(command.encode('utf-8'))
        except serial.SerialTimeoutException:
            print("Error: Write timeout occurred")
            return -2

        # Wait for initial confirmation ("R")
        try:
            initial_response = self.serial_com.read(size=2, timeout=self.msg_timeout_).decode('utf-8').strip()
            if len(initial_response) > 0:
                if initial_response[0] != 'R':
                    print(" MESSAGE: NOT RECEIVED. UNEXPECTED CONFIRMATION MESSAGE RECEIVED.")
                    print(" INFO   : Please, check that the version of the program is correct.")
                    return -3
        except serial.SerialTimeoutException:
            print(" MESSAGE: NOT RECEIVED. TIME OUT.")
            print(" INFO   : Please, check that the robot is ready to work.")
            return -5
        except serial.SerialException:
            print(" MESSAGE: NOT RECEIVED. COMMUNICATION ERROR.")
            print(" INFO   : Please, check that the RS-232 connection is properly set.")
            return -4

        # Wait for position confirmation ("F")
        try:
            pos_response = self.serial_com.read(size=2, timeout=self.pos_timeout_).decode('utf-8').strip()
            if len(pos_response) > 0:
                if pos_response[0] == 'F':
                    if verbose:
                        print(" POS OK : RECEIVED")
                        print("===============================================================")
                    return 0
                else:
                    print(" POS OK : NOT RECEIVED. UNEXPECTED CONFIRMATION MESSAGE RECEIVED.")
                    print(" INFO   : Please, check that the version of the program is correct.")
                    return -6
        except serial.SerialTimeoutException:
            print(" POS OK : NOT RECEIVED. TIME OUT.")
            print(" INFO   : May an error has occurred. Please, check that the robot is ready to work.")
            print("           If the robot moves very slowly, set a bigger position timeout.")
            return -1
        except serial.SerialException:
            print(" POS OK : NOT RECEIVED. COMMUNICATION ERROR.")
            print(" INFO   : Please, check that the RS-232 connection is properly set.")
            return -7

    def movePTP(self, pose, speed=15, tool=0, fp = 95, gripper=False, figure=5, verbose = False):
        if not self.isConnected():
            raise RuntimeError("Not connected to the robot")

        # Define the command
        command = f"1\r{pose[0]},{pose[1]},{pose[2]},{pose[3]},{pose[4]},{pose[5]},{figure},{speed},{tool},{fp}, {int(gripper)}\r"
        if verbose:
            print("\n====== Go To ===============================================================")
            print(f" POS    : {pose[0]}, {pose[1]}, {pose[2]}")
            print(f" ORIENT : {pose[3]}, {pose[4]}, {pose[5]}")
            print(f" SPEED  : {speed}")
            print(f" TOOL   : {tool}")
            print(f" FIGURE : {figure}")
            print(f" FP     : {fp}")
            print(f" GRIPPER: {'CLOSE' if gripper else 'OPEN'}")
            print("------------------------------------------------------------------------------")

        # Send the command
        try:
            self.serial_com.write(command.encode('utf-8'))
        except serial.SerialTimeoutException:
            print("Error: Write timeout occurred")
            return -2

        # Receive the initial response ("R")
        try:
            initial_response = self.serial_com.read(2).decode('utf-8').strip()
            if len(initial_response) >0:
                if initial_response[0] != 'R':
                    print("Error: Unexpected confirmation message received")
                    return -3
        except serial.SerialTimeoutException:
            print("Error: Timeout while waiting for initial response")
            return -5
        except serial.SerialException:
            print("Error: Communication error while waiting for initial response")
            return -4
        if verbose:
            print(" MESSAGE: RECEIVED")

        # Wait for position confirmation ("F")
        try:
            pos_response = self.serial_com.read(2).decode('utf-8').strip()
            if len(pos_response) >0:
                if pos_response[0] == 'F':
                    if verbose:
                        print(" POS OK : RECEIVED")
                        print("============================================================================")
                    return 0
                else:
                    print("Error: Unexpected position confirmation message received")
                    return -6
        except serial.SerialTimeoutException:
            print(" POS OK : NOT RECEIVED. TIME OUT.")
            print(" INFO   : The robot may be moving slowly. Increase the position timeout.")
            return -1
        except serial.SerialException:
            print(" POS OK : NOT RECEIVED. COMMUNICATION ERROR.")
            print(" INFO   : Check the RS-232 connection.")
            return -7

    def moveLine(self, pose, speed=15, tool=0, fp = 95, gripper=False, figure=5, verbose = False):
        if not self.isConnected():
            raise RuntimeError("Not connected to the robot")

        # Define the command
        command = f"5\r{pose[0]},{pose[1]},{pose[2]},{pose[3]},{pose[4]},{pose[5]},{figure},{speed},{tool},{fp}, {int(gripper)}\r"
        if verbose:
            print("\n====== Go To ===============================================================")
            print(f" POS    : {pose[0]}, {pose[1]}, {pose[2]}")
            print(f" ORIENT : {pose[3]}, {pose[4]}, {pose[5]}")
            print(f" SPEED  : {speed}")
            print(f" TOOL   : {tool}")
            print(f" FIGURE : {figure}")
            print(f" FP     : {fp}")
            print(f" GRIPPER: {'CLOSE' if gripper else 'OPEN'}")
            print("------------------------------------------------------------------------------")

        # Send the command
        try:
            self.serial_com.write(command.encode('utf-8'))
        except serial.SerialTimeoutException:
            print("Error: Write timeout occurred")
            return -2

        # Receive the initial response ("R")
        try:
            initial_response = self.serial_com.read(2).decode('utf-8').strip()
            if len(initial_response) >0:
                if initial_response[0] != 'R':
                    print("Error: Unexpected confirmation message received")
                    return -3
        except serial.SerialTimeoutException:
            print("Error: Timeout while waiting for initial response")
            return -5
        except serial.SerialException:
            print("Error: Communication error while waiting for initial response")
            return -4
        if verbose:
            print(" MESSAGE: RECEIVED")

        # Wait for position confirmation ("F")
        try:
            pos_response = self.serial_com.read(2).decode('utf-8').strip()
            if len(pos_response) >0:
                if pos_response[0] == 'F':
                    if verbose:
                        print(" POS OK : RECEIVED")
                        print("============================================================================")
                    return 0
                else:
                    print("Error: Unexpected position confirmation message received")
                    return -6
        except serial.SerialTimeoutException:
            print(" POS OK : NOT RECEIVED. TIME OUT.")
            print(" INFO   : The robot may be moving slowly. Increase the position timeout.")
            return -1
        except serial.SerialException:
            print(" POS OK : NOT RECEIVED. COMMUNICATION ERROR.")
            print(" INFO   : Check the RS-232 connection.")
            return -7