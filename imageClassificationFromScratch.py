import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, Rescaling
from tensorflow.keras.layers import Conv2D, BatchNormalization,  Activation, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, add, Dropout, Dense


# run the following command in the terminal (capital letter O not zero) to download the zip file
# curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip

# run the following command in the terminal to expand the zip file
# unzip -q kagglecatsanddogs_3367a.zip

# filter out corrupted images
# do this to clean up images before they are used to create a dataset
import os
num_skipped_type_int = 0

# give detail that this string looks like "Cat"
for folder_name_type_str in (
    "Cat",
    "Dog"
):

    # give detail that this string looks like "PetImages/Cat"
    folder_path_type_str = os.path.join(
        "PetImages",
        folder_name_type_str
    )

    # give detail that this string looks like "8691.jpg"
    for file_name_type_str in os.listdir( folder_path_type_str ):

        # give detail that this string looks like "PetImages/Cat/8691.jpg"
        file_path_type_str = os.path.join(
            folder_path_type_str,
            file_name_type_str
        )

        try:

            # find out what this BufferedReader object does and what it is used for?
            file_obj_type_BufferedReader = open(
                file_path_type_str,
                "rb"
            )

            # find out what this boolean does?
            # suggest that the peek tries to look into the file and if it can't produces an error
            is_jfif_type_bool = tf.compat.as_bytes( "JFIF" ) in file_obj_type_BufferedReader.peek( 10 )

        finally:

            # give a perspective that this closes the object defined in the try block
            file_obj_type_BufferedReader.close()

        if not is_jfif_type_bool:
            num_skipped_type_int += 1
            # delete corrupted image
            os.remove( file_path_type_str )

# print the number of deleted images
# mention that if 0 images have been deleted when beforehand it wasn't 0, it's because they were deleted
print( "Deleted %d images" % num_skipped_type_int )


# generate a dataset

image_size_type_tuple = (
    180,
    180
)

batch_size_type_int = 32

# split the dataset into 80% training and 20% validation by using the validation split, the subset and the random seed

train_ds_type_BatchDataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split = 0.2,
    subset = "training",
    seed = 1337, # fix a random seed for reproducibility purposes
    image_size = image_size_type_tuple,
    batch_size = batch_size_type_int
)

val_ds_type_BatchDataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split = 0.2,
    subset = "validation",
    seed = 1337,
    image_size = image_size_type_tuple,
    batch_size = batch_size_type_int
)


# visualise the data
import matplotlib.pyplot as plt

plt.figure( figsize = (
    10,
    10
) )

# get the number of batch sized specified amount (ie 32) of images and the corresponding labels as a Tensor object
# mention that the EagerTensor values would need to be converted to numpy to see values one expects
for images_type_EagerTensor, labels_type_EagerTensor in train_ds_type_BatchDataset.take( 1 ):
    for i_type_int in range( 9 ):

        # mention that the values go from 1 to the 3 x 3 = 9 for the subplot
        ax_type_AxesSubplot = plt.subplot(
            3,
            3,
            i_type_int + 1
        )

        # mention that the numpy array is unsigned
        # further mention that unsigned means no negative sign
        # emphasise then that uint8 starts from 0 and goes to 255 and is an integer
        plt.imshow( images_type_EagerTensor[ i_type_int ].numpy().astype( "uint8" ) )

        # attach the label value as the title for example in this case it is 1
        plt.title( int( labels_type_EagerTensor[ i_type_int ] ) )

        # remove the x and y axes values
        plt.axis( "off" )

    plt.show()

# make use of data augmentation
data_augmentation_type_Sequential = keras.Sequential(
    [
        RandomFlip( "horizontal" ),
        RandomRotation( 0.2 )
    ]
)

plt.figure( figsize = (
    10,
    10
))

# take a batch of images with their labels
for images_type_EagerTensor, _ in train_ds_type_BatchDataset.take(1):
    for i_type_int in range( 9 ):

        # augment all the original images using the data augmentation technique mentioned above
        augmented_images_type_EagerTensor = data_augmentation_type_Sequential( images_type_EagerTensor )

        ax_type_AxesSubplot = plt.subplot(
            3,
            3,
            i_type_int + 1
        )

        # this correspends to showing the result of applying data augmentation slight variants to the 0th image only
        plt.imshow( augmented_images_type_EagerTensor[ 1 ].numpy().astype( "uint8" ) )

        plt.axis( "off" )

    plt.show()

try:
    # Show two methods of preprocessing the dataset
    inputs = keras.Input( shape = input_shape )
    x = data_augmentation_type_Sequential( inputs )
    x = Rescaling( 1. / 255 )( x )
    ... # Rest of the model
except Exception:
    print("Testing two ways of preprocessing the dataset to make use of data augmentation")

try:
    augmented_train_ds = train_ds_type_BatchDataset.map( lambda x,y: (

        data_augmentation_type_Sequential(
            x,
            training= True
        ),

        y
    ) )
except Exception:
    print( "this is another method for data augmentation " )

train_ds = train_ds_type_BatchDataset.prefetch( buffer_size = 32 )
val_ds = val_ds_type_BatchDataset.prefetch( buffer_size = 32 )

def make_model(
        input_shape,
        num_classes
):
    inputs = keras.Input( shape = input_shape )
    # Image augmentation block
    # use the first method for data augmentation, making it a part of the model
    x = data_augmentation_type_Sequential( inputs )


    # Entry block
    x = Rescaling( 1.0 / 255 )( x )

    x = Conv2D(
        32,
        3,
        strides = 2,
        padding = "same"
    )( x )

    x = BatchNormalization()( x )
    x = Activation( "relu" )( x )

    x = Conv2D(
        64,
        3,
        padding = "same"
    )( x )

    x = BatchNormalization()( x )
    x = Activation( "relu" )( x )

    previous_block_activation = x # Set aside residual

    for size in [
        128,
        256,
        512,
        728
    ]:
        x = Activation( "relu" )( x )

        x = SeparableConv2D(
            size,
            3,
            padding = "same"
        )( x )

        x = BatchNormalization()( x )

        x = Activation( "relu" )( x )

        x = SeparableConv2D(
            size,
            3,
            padding = "same"
        )( x )

        x = BatchNormalization()( x )

        x = MaxPooling2D(
            3,
            strides = 2,
            padding = "same"
        )( x )

        # Project residual
        residual = Conv2D(
            size,
            1,
            strides = 2,
            padding = "same"
        )( previous_block_activation )

        x = add( [
            x,
            residual
        ] ) # Add back (the) residual

        previous_block_activation = x # Set aside next residual

    x = SeparableConv2D(
        1024,
        3,
        padding = "same"
    )( x )

    x = BatchNormalization()( x )
    x = Activation( "relu" )( x )

    x = GlobalAveragePooling2D()( x )

    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = Dropout( 0.5 )( x )
    outputs = Dense(
        units,
        activation = activation
    )( x )

    return keras.Model(
        inputs,
        outputs
    )

model = make_model(
    input_shape = image_size_type_tuple + (3, ),
    num_classes = 2
)

keras.utils.plot_model(
    model,
    show_shapes = True
)

epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint( "save_at_{epoch}.h5" ),
]

model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

model.fit(
    train_ds_type_BatchDataset,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = val_ds_type_BatchDataset
)

# run inference on new data

img = keras.preprocessing.image.load_img(
    "PetImages/Cat/6779.jpg",
    target_size = image_size_type_tuple
)

img_array = keras.preprocessing.image.img_to_array( img )
img_array = tf.expand_dims( img_array, 0 ) # Create batch axis

predictions = model.predict( img_array )
score = predictions[ 0 ]

print(
    "This image is %.2f percent cat and %.2f percent dog."
    % ( 100 * (1-score), 100 * score )
)