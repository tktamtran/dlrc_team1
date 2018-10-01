import py_at_broker as pab
import numpy as np
import time, math

# to open simulator, cd ~/git/sl/build/sl_panda  $ ./xpanda
# to avoid "Address broker could not acceppt the offered signal franka_target_pos.", restart $ addr_broker

# virtual limits are in file ~/git/sl/sl_panda/src/pyctl/pyctl_traj.h at the very end
#            const double vWallMin[3] = {0.28, -0.78, 0.02};
#            const double vWallMax[3] = {0.82,  0.78, 1.08};

# remember to st 6 in xpanda before/during code execution

broker = pab.broker("tcp://localhost:51468")
broker.register_signal("franka_target_pos", pab.MsgType.target_pos)

def gen_msg(broker, fnumber, new_pos):
    msg = pab.target_pos_msg()
    msg.set_timestamp(time.time())
    msg.set_ctrl_t(0)
    msg.set_fnumber(fnumber)
    msg.set_pos(new_pos)
    msg.set_time_to_go(0.2)
    broker.send_msg("franka_target_pos", msg)

# zig-zagging around a box
f = 0
while(False):
    for i in np.arange(0.5,0.7,0.01):
        for j in np.arange(0.5,0.7,0.01):
            for k in np.arange(0.5,0.7,0.01):
                gen_msg(broker, f, np.array([i,j,k]))
                print(i,j,k)
                f+=1
                time.sleep(0.02)


# spiral like a churro
up = True
z = 0.2
c = 0.001                
while(False):
    for i in np.arange(0,1,0.075):
        # infinite loop
        #x = 0.5+0.1*math.cos(i * 2*math.pi)
        #y = 0.5+0.1*math.sin(i * 2*2*math.pi)
        #z = 0.5+0.1*math.sin(i * 2*math.pi)
        # spiral
        #xcenter = 0.3*math.cos(c*2*math.pi)
        #ycenter = 0.3*math.sin(c*2*math.pi)
        #x = xcenter+0.1*math.cos(i * 2*math.pi)
        #y = ycenter+0.1*math.sin(i * 2*math.pi)
        x = 0.5+0.1*math.cos(i * 2*math.pi)
        y = 0.5+0.1*math.sin(i * 2*math.pi)
        if(up):
            z += 0.005
        else:
            z -= 0.005
        if(z > 0.7):
            up = False
        if(z < 0.2):
            up = True
        print(x,y,z)
        #c += 0.001
        gen_msg(broker, f, np.array([x,y,z]))
        f+=1
        time.sleep(0.5)
        
            
            
# moving to specific points
while(True):
    pos = np.array([0.5,0.5,0.3])
    gen_msg(broker, f,pos)
    time.sleep(2)

    for i in range(10): # go up and down 10 times
        for f in np.arange(30,70,1):
            #gen_msg(broker, f, np.array([0.3, 0.4, 0.00]))
            gen_msg(broker, f, np.array([0.50,0.50,f/100]))
            print(f/100)
            time.sleep(0.2)

        for f in np.arange(70, 30, -1):
            gen_msg(broker, f, np.array([0.50, 0.50, f/100]))
            print(f/100)
            time.sleep(0.2)
