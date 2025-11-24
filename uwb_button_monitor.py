import threading
import time
import os
from unitree_sdk2py.idl.unitree_go.msg.dds_ import UwbState_

class UwbButtonMonitor:
    def __init__(self, state_manager, on_x_pressed_callback):
        self.state_manager = state_manager
        self.on_x_pressed_callback = on_x_pressed_callback
        self.last_buttons_state = 0

    def get_callback(self):
        def uwb_callback(msg: UwbState_):
            self.state_manager.update_state(msg)
            current_buttons = msg.buttons
            changed = current_buttons ^ self.last_buttons_state
            if changed == 0:
                return

            BUTTON_X_MASK = 1 << 2  # X button = bit 2

            if changed & BUTTON_X_MASK:
                if current_buttons & BUTTON_X_MASK:
                    print("[UWB] X button pressed — initiating shutdown.")

                    def shutdown():
                        self.on_x_pressed_callback()
                        time.sleep(0.1)
                        os._exit(0)


                    threading.Thread(target=shutdown, daemon=True).start()

            self.last_buttons_state = current_buttons


        return uwb_callback
