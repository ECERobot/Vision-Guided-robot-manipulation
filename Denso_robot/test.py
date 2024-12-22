import numpy as np
from DensoRobotControl import DensoRobotControl



if __name__ == "__main__":
    try:
        robot = DensoRobotControl(port="COM4", baud_rate=19200)
        if robot.isConnected():
            robot.setTimeout(msg_time_ms=100, pos_time_ms=1500000)
            robot.getPosition(verbose = False)
            input("enter")
            # robot.moveLine(pose=[350, 20, 570, -180,0,-90], speed=15, tool=0,verbose=False)
            robot.moveLine(pose=[513, 195, 511, 180, 4 ,-90], speed=15, tool=0,verbose=False)
    except RuntimeError as e:
        print(e)