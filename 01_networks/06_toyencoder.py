# inputs to the net:    joint angles
#                       pixel_idx
#                       possibly cam_origin

# outputs of the net:   mean (mu) of sensor reading / wcs point
#                       variance/log variance/stddev (sigma) of sensor reading / wcs point
#                       choice of representing inputs as sensor reading or wcs
#                       is a hyperparameter in itself


# net design:   input layer that accepts the above specified inputs
#               hidden layer(s) (exact config is hyperparameter)
#               output layer that gives the expected mu and sigma

# loss function:    compares output of the net at the current stage to the known
#                   sensor readings / wcs points
#                   which loss function to choose exactly?
#                   should try KL, vaserstein/earth-mover and MSE
#                   (so choice of loss is another hyperparameter)


from keras.models import Model
from keras.layers import Input, Dense

# the input layer has the shape:
# 3
input_layer =