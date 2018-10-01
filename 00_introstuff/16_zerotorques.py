import dlrc_control as ctrl


broker = ctrl.initialize()

while(True):
    ctrl.set_zero_torques(broker)