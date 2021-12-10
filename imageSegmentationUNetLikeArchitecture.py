# I REALLY LIKE THE DATA I AM SEEING SO FAR
# DOWNLOAD THE DATA (Oxford Pets Dataset)

# type the following in the terminal

# curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

# the following expands the compressed tar.gz files
# tar -xf images.tar.gz
# tar -xf annotations.tar.gz

# PREPARE PATHS OF INPUT IMAGES AND TARGET SEGMENTATION MASKS

from os.path import join
from os import listdir

input_dir_type_str = "images/"
target_dir_type_str = "annotations/trimaps/"

img_size_type_tuple = (
    160,
    160
)

num_class_type_int = 3
batch_size_type_int = 3

# mention that the list has 7390 elements in it
input_img_paths_type_list = sorted(
    [

        join(
            input_dir_type_str,
            filename_type_str
        )

        for filename_type_str in listdir( input_dir_type_str )
        if filename_type_str.endswith( ".jpg" )
    ]
)

# mention that the list has 7390 elements in it
target_img_paths_type_list = sorted(
    [
        join(
            target_dir_type_str,
            filename_type_str
        )

        for filename_type_str in listdir( target_dir_type_str )
        # check that it it not for example ".png" or ".fffffffff.png"
        if filename_type_str.endswith( ".png" ) and not filename_type_str.startswith( "." )
    ]
)

print(
    "Number of samples:",
    len( input_img_paths_type_list)
)

for input_path_type_str, target_path_type_str in zip(
    input_img_paths_type_list[:10],
    target_img_paths_type_list[:10]
):

    print(
        input_path_type_str,
        "|",
        target_path_type_str
    )


# WHAT DOES ONE INPUT IMAGE AND THE CORRESPONDING SEGMENTATION MASK LOOK LIKE?

from IPython.display import Image,display
from tensorflow.keras.preprocessing.image import load_img
from PIL.ImageOps import autocontrast
from matplotlib.pyplot import imshow, show

chosen_index_type_int = 9

# Display input image #7
example_image_type_Image = load_img(  input_img_paths_type_list[ chosen_index_type_int ] )
imshow( example_image_type_Image )
show()

# Display auto-contrast version of corresponding target (per-pixel categories)
img_type_Image = autocontrast( load_img( target_img_paths_type_list[ chosen_index_type_int ] ) )
imshow( img_type_Image )
show()

# PREPARE SEQUENCE CLASS TO LOAD AND VECTORIZE BATCHES OF DATA

from tensorflow.keras.utils import Sequence
from numpy import zeros
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img

class OxfordPets( Sequence):
    """ Helper to iterate over the data ( as Numpy arrays) """

    def __init__(
            self,
            batch_size,
            img_size,
            input_img_paths,
            target_img_paths
    ):

        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__( self ):
        return len( self.target_img_paths ) // self.batch_size

    def __getitem__(
            self,
            idx
    ):

        """ Returns tuple (input, target) corresponding to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[ i : ( i + self.batch_size ) ]
        batch_target_img_paths = self.target_img_paths[ i : ( i + self.batch_size ) ]

        x = zeros(
            ( self.batch_size, ) + self.img_size + ( 3, ),
            dtype = "float32"
        )

        for j, path in enumerate( batch_input_img_paths ):

            img = load_img(
                path,
                target_size = self.img_size
            )

            x[ j ] = img

        y = zeros(
            ( self.batch_size, ) + self.img_size + ( 1, ),
            dtype = "uint8"
        )

        for j, path in enumerate( batch_target_img_paths ):

            img = load_img(
                path,
                target_size = self.img_size,
                color_mode = "grayscale"
            )

            y[ j ]= expand_dims(
                img,
                2
            )

            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[ j ] -= 1

        return x, y


# PREPARE U-NET XCEPTION-STYLE MODEL

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D, MaxPooling2D, add, Conv2DTranspose, UpSampling2D
from tensorflow.keras.backend import clear_session


def get_model(
        img_size,
        num_classes
):

    inputs = Input( shape = img_size + ( 3, ) )

    ### [ First half of the network: downsampling inputs ] ###

    # Entry block

    x = Conv2D(
        32,
        3,
        strides = 2,
        padding = "same"
    )( inputs )

    x = BatchNormalization()( x )
    x = Activation( "relu" )( x )

    previous_block_activation = x # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters_type_int in [
        64,
        128,
        256
    ]:
        pass

    pass

