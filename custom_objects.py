from tensorflow.keras import layers
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers import  RandomRotation, RandomZoom, Layer


class CustomDataAugmentation(Layer):
    """
    Custom data augmentation layer that combines multiple augmentation techniques.
    This layer applies random rotation and random zoom to input images.
    """
    
    def __init__(
        self, 
        rotation_factor=0.2, 
        zoom_height_factor=0.2, 
        zoom_width_factor=0.2,
        seed=None,
        fill_mode='reflect',
        interpolation='bilinear',
        fill_value=0.0,
        name="custom_augmentation",
        **kwargs
    ):
        """
        Initialize the data augmentation layer.
        
        Args:
            rotation_factor: Maximum rotation angle in radians or degrees.
                             If float, interpreted as radians; if int, interpreted as degrees.
            zoom_height_factor: Range for random height zoom. 
                                For a factor of 0.2, the zoomed area will be between 80% to 120% of original size.
            zoom_width_factor: Range for random width zoom.
                               If None, same as height_factor.
            seed: Random seed for reproducibility.
            fill_mode: Points outside boundaries are filled according to the given mode
                       (one of {'constant', 'reflect', 'wrap', 'nearest'}).
            interpolation: Interpolation method used to fill in new pixels.
                           One of {'nearest', 'bilinear'}.
            fill_value: Value used for points outside boundaries when fill_mode='constant'.
            name: Name of the layer.
        """
        super(CustomDataAugmentation, self).__init__(name=name, **kwargs)
        
        # Create the individual augmentation layers
        self.random_rotation = RandomRotation(
            rotation_factor,
            fill_mode=fill_mode,
            interpolation=interpolation,
            seed=seed,
            fill_value=fill_value
        )
        
        self.random_zoom = RandomZoom(
            height_factor=zoom_height_factor,
            width_factor=zoom_width_factor,
            fill_mode=fill_mode,
            interpolation=interpolation,
            seed=seed,
            fill_value=fill_value
        )
    
    def call(self, inputs, training=None):
        """
        Apply the augmentation to input images.
        
        Args:
            inputs: Input tensor, usually images.
            training: Boolean indicating whether the layer should behave in
                      training mode (applying augmentation) or inference mode (identity).
        
        Returns:
            Augmented images if training=True, original images otherwise.
        """
        if training is None:
            training = tf.keras.backend.learning_phase()
            
        if training:
            x = self.random_rotation(inputs)
            x = self.random_zoom(x)
            return x
        else:
            return inputs
    
    def compute_output_shape(self, input_shape):
        """The output shape is the same as the input shape."""
        return input_shape
    
    def get_config(self):
        """Return the configuration for serialization."""
        config = super(CustomDataAugmentation, self).get_config()
        # Add the configuration of the individual augmentation layers
        config.update({
            "rotation_factor": self.random_rotation.factor,
            "zoom_height_factor": self.random_zoom.height_factor,
            "zoom_width_factor": self.random_zoom.width_factor,
            "fill_mode": self.random_rotation.fill_mode,
            "interpolation": self.random_rotation.interpolation,
            "fill_value": self.random_rotation.fill_value,
            "seed": self.random_rotation.seed,
        })
        return config
    

# Custom preprocess Layer for easy serialization
class ResNetPreprocessingLayer(layers.Layer):
    """Custom layer that applies ResNet preprocessing to input images."""
    
    def __init__(self, name="resnet_preprocessing", **kwargs):
        super(ResNetPreprocessingLayer, self).__init__(name=name, **kwargs)
    
    def call(self, inputs):
        """Apply ResNet preprocessing to the input tensor."""
        return preprocess_input(inputs)
    
    def compute_output_shape(self, input_shape):
        """The output shape is the same as the input shape."""
        return input_shape
    
    def get_config(self):
        """Return the config dictionary for serialization."""
        config = super(ResNetPreprocessingLayer, self).get_config()
        return config