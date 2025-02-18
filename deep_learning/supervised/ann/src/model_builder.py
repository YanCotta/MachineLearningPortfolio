import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from typing import List, Optional, Union, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuralNetworkBuilder:
    """
    A class to build and configure neural network architectures.
    
    Attributes:
        input_dim (int): Input dimension for the network
        architecture (dict): Network architecture configuration
        model (tf.keras.Model): The constructed neural network model
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize the model builder.
        
        Args:
            input_dim (int): Dimension of input features
        """
        self.input_dim = input_dim
        self.model = None
        
    def build_model(self,
                hidden_layers: List[int],
                dropout_rates: Optional[List[float]] = None,
                activation: str = 'relu',
                output_activation: str = 'sigmoid',
                use_batch_norm: bool = True,
                kernel_regularizer: Optional[Dict[str, float]] = None) -> Model:
        """
        Build a neural network with specified architecture.
        
        Args:
            hidden_layers (List[int]): Number of neurons in each hidden layer
            dropout_rates (Optional[List[float]]): Dropout rate for each layer
            activation (str): Activation function for hidden layers
            output_activation (str): Activation function for output layer
            use_batch_norm (bool): Whether to use batch normalization
            kernel_regularizer (Optional[Dict[str, float]]): L1/L2 regularization parameters
            
        Returns:
            tf.keras.Model: Constructed neural network model
        """
        logger.info("Building neural network model")
        
        model = Sequential()
        
        # Input layer
        first_layer = True
        
        # Add hidden layers
        for i, neurons in enumerate(hidden_layers):
            # Add Dense layer
            if first_layer:
                model.add(Dense(neurons, 
                            activation=activation,
                            input_dim=self.input_dim,
                            kernel_regularizer=l1_l2(**kernel_regularizer) if kernel_regularizer else None))
                first_layer = False
            else:
                model.add(Dense(neurons, 
                            activation=activation,
                            kernel_regularizer=l1_l2(**kernel_regularizer) if kernel_regularizer else None))
            
            # Add BatchNormalization if specified
            if use_batch_norm:
                model.add(BatchNormalization())
            
            # Add Dropout if specified
            if dropout_rates and i < len(dropout_rates):
                model.add(Dropout(dropout_rates[i]))
        
        # Output layer
        model.add(Dense(1, activation=output_activation))
        
        logger.info(f"Model built successfully with {len(hidden_layers)} hidden layers")
        self.model = model
        return model
    
    def compile_model(self,
                    optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
                    loss: str = 'binary_crossentropy',
                    metrics: List[str] = ['accuracy']) -> None:
        """
        Compile the neural network model.
        
        Args:
            optimizer (Union[str, tf.keras.optimizers.Optimizer]): Optimizer for training
            loss (str): Loss function
            metrics (List[str]): Metrics to track during training
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        logger.info(f"Compiling model with {optimizer} optimizer and {loss} loss")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def get_model_summary(self) -> str:
        """
        Get a string representation of the model architecture.
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            raise ValueError("Model must be built first")
            
        # Create a string buffer to capture the summary
        from io import StringIO
        summary_buffer = StringIO()
        self.model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
        return summary_buffer.getvalue()
    
    @staticmethod
    def create_default_model(input_dim: int) -> Model:
        """
        Create a default model with recommended architecture.
        
        Args:
            input_dim (int): Input dimension
            
        Returns:
            tf.keras.Model: Default neural network model
        """
        builder = NeuralNetworkBuilder(input_dim)
        model = builder.build_model(
            hidden_layers=[32, 16, 8],
            dropout_rates=[0.3, 0.2, 0.1],
            use_batch_norm=True,
            kernel_regularizer={'l1': 1e-5, 'l2': 1e-4}
        )
        builder.compile_model()
        return model
