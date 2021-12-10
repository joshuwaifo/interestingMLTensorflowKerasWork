# SETUP / IMPORT THE NEEDED LIBRARIES

from tensorflow.keras.datasets.mnist import load_data
from numpy import expand_dims
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

# PREPARE THE DATA

# Model / data parameters
num_classes_type_int = 10
input_shape_type_tuple = ( 28, 28, 1 )

# the data, split between train and test sets
( x_train_type_ndarray, y_train_type_ndarray ), ( x_test_type_ndarray, y_test_type_ndarray ) = load_data()

# Scale images to the [ 0, 1 ] range
# each of the following variables are numpy arrays whose values are between 0 and 1
x_train_type_ndarray = x_train_type_ndarray.astype( "float32" ) / 255
x_test_type_ndarray = x_test_type_ndarray.astype( "float32" ) / 255


# Make sure images have shape ( 28, 28, 1 )
x_train_type_ndarray = expand_dims( x_train_type_ndarray, -1 )
x_test_type_ndarray = expand_dims( x_test_type_ndarray, -1 )

print(
    "x_train_type_ndarray shape:",
    x_train_type_ndarray.shape
)

print(
    x_train_type_ndarray.shape[ 0 ],
    "train samples"
)

print(
    x_test_type_ndarray.shape[ 0 ],
    "test samples"
)

# mention that before the to_categorical each variable is a numpy array whpse values are 0, 1, 2, 3, ..., 9
# convert class vectors to binary class matrices
y_train_type_ndarray = to_categorical( y_train_type_ndarray, num_classes_type_int )
y_test_type_ndarray = to_categorical( y_test_type_ndarray, num_classes_type_int )
# mention that after the to_categorical each variable ie y_train_type_ndarray is a one hot encoding of the 10 classes

# BUILD THE MODEL

model_type_Sequential = Sequential( [
    Input( shape = input_shape_type_tuple ),

    Conv2D(
        32,

        kernel_size = (
            3,
            3
        ),

        activation = "relu"

    ),

    MaxPooling2D(

        pool_size = (
            2,
            2
        )

    ),

    Conv2D(
        64,

        kernel_size = (
            3,
            3
        ),

        activation = "relu"

    ),

    MaxPooling2D(

        pool_size = (
            2,
            2
        )

    ),

    Flatten(),
    Dropout( 0.5 ),

    Dense(
        num_classes_type_int,
        activation = "softmax"
    )

] )

model_type_Sequential.summary()

# TRAIN THE MODEL

# batch_size_type_int = 128
# epochs_type_int = 15
batch_size_type_int = 16
epochs_type_int = 1

model_type_Sequential.compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = [ "accuracy" ]
)

model_type_Sequential.fit(
    x_train_type_ndarray,
    y_train_type_ndarray,
    batch_size = batch_size_type_int,
    epochs = epochs_type_int,
    validation_split = 0.1
)


# EVALUATE THE TRAINED MODEL

score_type_list = model_type_Sequential.evaluate(
    x_test_type_ndarray,
    y_test_type_ndarray,
    verbose = 0
)

print(
    "Test loss:",
    score_type_list[ 0 ]
)

print(
    "Test accuracy:",
    score_type_list[ 1 ]
)

# Mention some ways to improve this, maybe exploring the data distribution split and deciding to use a difference metric like precision or recall
# Mention that precision is from the perspective of the model's prediction whereas recall is from the perspective of the actual values
