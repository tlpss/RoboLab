import threading
import time
from collections import deque

import loguru
import pyspacemouse


logger = loguru.logger


class SpaceMouseAgent:
    """spacemouse teleop.
    provides a delta to the current pose of the robot and gripper.
    [x,y,z,rx,ry,rz,gripper]
    """

    ACTION_SPEC = None

    def __init__(self, deadzone=0.1, translation_scale=0.02, rotation_scale=0.02, gripper_step_size=0.01):
        """
        Args:
            deadzone: if input of spacemouse is below this value, it is set to 0. makes it easier to teleop single axis.
            translation_scale: scale for translation, this is the real-world displacement per spacemouse action in meters.
            rotation_scale: scale for rotation, this is the real-world rotation per spacemouse action in radians.
            gripper_step_size: step size for gripper, this is the real-world displacement per spacemouse action in meters.
        """
        self.state_buffer = deque(maxlen=10)  # Use deque as a rolling buffer
        self.running = True
        self.deadzone = deadzone
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.gripper_step_size = gripper_step_size
        self.device = None

        # separate thread needed for continuous reading of SpaceMouse
        # cf. https://github.com/wuphilipp/gello_software/blob/main/gello/agents/spacemouse_agent.py
        #
        self.thread = threading.Thread(target=self._spacemouse_thread)
        self.thread.daemon = True
        self.thread.start()

    def _spacemouse_thread(self):
        try:
            self.device = pyspacemouse.open()
            logger.info("SpaceMouse connected successfully!")
        except Exception as e:
            logger.error(f"Could not open SpaceMouse: {e}")
            return

        while self.running:
            try:
                state = self.device.read()
                if state is not None:
                    self.state_buffer.append(state)
                    logger.debug(f"Buffered state: {state}")
            except Exception as e:
                logger.error(f"Error reading SpaceMouse: {e}")
                break
        
        if self.device:
            self.device.close()

    def get_action(self, observation=None):
        del observation

        if not self.state_buffer:  # check if buffer is empty.
            logger.warning("SpaceMouse buffer is empty.")
            return [0, 0, 0, 0, 0, 0, 0]

        state = self.state_buffer.pop()

        # do a conversion from the spacemouse coordinate frame to a different coordinate frame that makes teleop more intuitive
        # for the orientations. each twist on the axis corresponds now to how you want the robot to twist as well.
        roll, pitch, yaw = -state.pitch, state.roll, -state.yaw
        rot = [roll, pitch, yaw]

        pos = [state.x, state.y, state.z]

        for i in range(3):
            if abs(pos[i]) < self.deadzone:
                pos[i] = 0
            else:
                pos[i] *= self.translation_scale

        for i in range(3):
            if abs(rot[i]) < self.deadzone:
                rot[i] = 0
            else:
                rot[i] *= self.rotation_scale

        gripper_action = 0
        if state.buttons[1]:
            gripper_action = -self.gripper_step_size
        elif state.buttons[0]:
            gripper_action = self.gripper_step_size

        return [*pos, *rot, gripper_action]

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()


if __name__ == "__main__":

    # check if spacemouse can be opened, otherwise print error and exit.
    devices = pyspacemouse.get_connected_devices()
    print(f"Connected SpaceMouse devices: {devices}")

    try:
        agent = SpaceMouseAgent()  # set buffer size.
        while True:
            action = agent.get_action()
            print(f"Action: {action}")
            time.sleep(0.05)
    except ValueError as e:
        print(e)
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        if "agent" in locals():
            agent.close()

    