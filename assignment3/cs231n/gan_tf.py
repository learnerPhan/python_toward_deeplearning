import numpy as np
import tensorflow as tf

NOISE_DIM = 96

def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.
    
    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU
    
    Returns:
    TensorFlow Tensor with the same shape as x
    """
    # TODO: implement leaky ReLU
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pos = tf.maximum(x, 0.0)
    neg = alpha*tf.minimum(x, 0.0)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return pos + neg
    
def sample_noise(batch_size, dim, seed=None):
    """Generate random uniform noise from -1 to 1.
    
    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the noise to generate
    
    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    if seed is not None:
        tf.random.set_seed(seed)
    # TODO: sample and return noise
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    noise = tf.random.uniform(shape=(batch_size, dim), minval=-1, maxval=1, dtype=tf.dtypes.float32)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return noise
    
def discriminator(seed=None):
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    if seed is not None:
        tf.random.set_seed(seed)

    model = None
    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: tf.keras.models.Sequential might be helpful.                         #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    """
    Reminder how to build model:
    1, Create layers list with the help of tf.keras.layers
        layers = [ tf.keras.layers.xx(params),
                   layer2,
                   ...
                 ]
    2, Build model with the above layers :
        model = tf.keras.Sequential(layers)
    """

    # first try : it works
    # So I need to soecify input_shape to solve error no weights
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=256, input_shape=(784,), activation=leaky_relu,
                                    use_bias=True))
    model.add(tf.keras.layers.Dense(units=256, input_shape=(256,), activation=leaky_relu,
                                    use_bias=True))
    model.add(tf.keras.layers.Dense(units=1, use_bias=True))
    """
    # second try : it works
    # still need input_shape
    layers = [tf.keras.layers.Dense(units=256, input_shape=(784,), activation=leaky_relu,
                                    use_bias=True),
              tf.keras.layers.Dense(units=256, input_shape=(256,), activation=leaky_relu,
                                    use_bias=True),
              tf.keras.layers.Dense(units=1, use_bias=True)
             ]

    model = tf.keras.Sequential(layers)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model

def generator(noise_dim=NOISE_DIM, seed=None):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """

    if seed is not None:
        tf.random.set_seed(seed)
    model = None
    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: tf.keras.models.Sequential might be helpful.                         #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    layers = [tf.keras.layers.Dense(units=1024, input_shape=(noise_dim,), activation='relu'),
              tf.keras.layers.Dense(units=1024, input_shape=(1024,), activation='relu'),
              tf.keras.layers.Dense(units=784, activation=tf.nn.tanh)
             ]

    model = tf.keras.Sequential(layers)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: Tensor of shape (N, 1) giving scores for the real data.
    - logits_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Returns:
    - loss: Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # My answer which make use of BinaryCrossentropy as suggested but wrong due to my not know how to use
    """
    # generate labels : real = ones, fake = zeros
    label_real = tf.ones_like(logits_real)
    label_fake = tf.zeros_like(logits_fake)

    # caculate losses using tf.keras.losses.BinaryCrossentropy
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) 
    loss_real = bce(logits_real, label_real)
    loss_fake = bce(logits_fake, label_fake)

    loss = tf.reduce_mean(loss_real) + tf.reduce_mean(loss_fake)

    """

    # stolen idea of 
    # https://github.com/jariasf/CS231n/blob/master/assignment3/GANs-TensorFlow.ipynb
    Dx = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real))
    DGx = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake))
    # Gx = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake))
    loss = tf.reduce_mean(Dx) + tf.reduce_mean(DGx)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    label_fake = tf.ones_like(logits_fake)

    gen = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=label_fake)
    loss = tf.reduce_mean(gen)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss


def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.
    
    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)
    
    Returns:
    - D_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    - G_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    """
    # TODO: create an AdamOptimizer for D_solver and G_solver
    D_solver = None
    G_solver = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    D_solver = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=beta1
               )

    G_solver = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=beta1
               )

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return D_solver, G_solver

def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: Tensor of shape (N, 1) giving scores for the real data.
    - scores_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Outputs:
    - loss: A Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 0.5*tf.reduce_mean((scores_real-1)**2) + 0.5*tf.reduce_mean((scores_fake)**2)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Outputs:
    - loss: A Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 0.5*tf.reduce_mean((scores_fake-1)**2)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def dc_discriminator():
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    model = None
    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: tf.keras.models.Sequential might be helpful.                         #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    """
    Reminder :
    1. How to build model
       a. Build layers with tf.keras.layers.X
           E.g : layer_conv = tf.keras.layers.Conv2D(parameters)
       b. Build model with tf.keras.Sequential(layers)

    2. Common layers
        2.1 Conv2D + relu
            tf.keras.layers.Conv2D(
              filters=32,
              kernel_size=(5,5),
              activation="relu"
            )

        2.2 Faltten
            tf.keras.layers.Flatten()

        2.3 Fully connected + relu
            tf.keras.layers.Dense(
              units=32,
              activation = "relu",
              use_bias=True
            )

    3. Our input shape is (N,H,W,C) = (N,28,28,1)
    """
    # build model
    model = tf.keras.Sequential()
    # make layer in architecture then add each in model
    reshape =   tf.keras.layers.Reshape(
                  input_shape=(784,),
                  target_shape=(28,28,1)
                )
    model.add(reshape)

    # input_shape = (28,28,1)
    conv1_relu = tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=5,
                    activation=leaky_relu,
                    strides=1,
                    padding='valid',
                    use_bias=True,
                    bias_initializer='zeros',
                    # input_shape=input_shape
                )
    model.add(conv1_relu)

    maxpool1 =   tf.keras.layers.MaxPool2D(
                    pool_size=2,
                    strides=2,
                    padding='valid'
                )
    model.add(maxpool1)
 
    conv2_relu = tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=5,
                    activation=leaky_relu,
                    strides=1,
                    padding='valid',
                    use_bias=True,
                    bias_initializer='zeros'
                )
    model.add(conv2_relu)

    maxpool2 =   tf.keras.layers.MaxPool2D(
                    pool_size=2,
                    strides=2,
                    padding='valid'
                )
    model.add(maxpool2)

    flat = tf.keras.layers.Flatten()
    model.add(flat)

    fc1_relu =   tf.keras.layers.Dense(
                    units=4*4*64,
                    activation=leaky_relu,
                    use_bias=True,
                    bias_initializer='zeros'
                )
    model.add(fc1_relu)

    fc2 =   tf.keras.layers.Dense(
                    units=1,
                    use_bias=True,
                    bias_initializer='zeros'
                )
    model.add(fc2)

    # for debug
    # model.summary()

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model


def dc_generator(noise_dim=NOISE_DIM):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    model = tf.keras.models.Sequential()
    # TODO: implement architecture
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    """
    Reminder:
    1. Architecture
        Fully connected with output size 1024
        ReLU
        BatchNorm
        Fully connected with output size 7 x 7 x 128
        ReLU
        BatchNorm
        Resize into Image Tensor of size 7, 7, 128
        Conv2D^T (transpose): 64 filters of 4x4, stride 2
        ReLU
        BatchNorm
        Conv2d^T (transpose): 1 filter of 4x4, stride 2
        TanH
    """
    # build model
    model = tf.keras.Sequential()

    # add layers
    fc1_relu =  tf.keras.layers.Dense(
                  units=1024,
                  use_bias=True,
                  activation='relu',
                  input_shape=(noise_dim,)
                )
    model.add(fc1_relu)

    bn1 =   tf.keras.layers.BatchNormalization()
    model.add(bn1)

    fc2_relu =  tf.keras.layers.Dense(
                  units=7*7*128,
                  use_bias=True,
                  activation='relu'
                )
    model.add(fc2_relu)

    bn2 =   tf.keras.layers.BatchNormalization()
    model.add(bn2)

    reshape = tf.keras.layers.Reshape(target_shape=(7,7,128))
    model.add(reshape)

    convT1_relu =   tf.keras.layers.Conv2DTranspose(
                      filters=64,
                      kernel_size=4,
                      strides=2,
                      padding='same',
                      use_bias=True,
                      activation='relu'
                    )
    model.add(convT1_relu)

    bn3 = tf.keras.layers.BatchNormalization()
    model.add(bn3)

    convT2_relu =   tf.keras.layers.Conv2DTranspose(
                      filters=1,
                      kernel_size=4,
                      strides=2,
                      padding='same',
                      activation='tanh'
                    )
    model.add(convT2_relu)

    # for debug
    # model.summary()

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model


# a giant helper function
def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss,\
              show_every=250, print_every=20, batch_size=128, num_epochs=10, noise_size=96):
    """Train a GAN for a certain number of epochs.
    
    Inputs:
    - D: Discriminator model
    - G: Generator model
    - D_solver: an Optimizer for Discriminator
    - G_solver: an Optimizer for Generator
    - generator_loss: Generator loss
    - discriminator_loss: Discriminator loss
    Returns:
        Nothing
    """
    mnist = MNIST(batch_size=batch_size, shuffle=True)
    
    iter_count = 0
    images = []
    for epoch in range(num_epochs):
        for (x, _) in mnist:
            with tf.GradientTape() as tape:
                real_data = x
                logits_real = D(preprocess_img(real_data))

                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)
                logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))

                d_total_error = discriminator_loss(logits_real, logits_fake)
                d_gradients = tape.gradient(d_total_error, D.trainable_variables)      
                D_solver.apply_gradients(zip(d_gradients, D.trainable_variables))
            
            with tf.GradientTape() as tape:
                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)

                gen_logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))
                g_error = generator_loss(gen_logits_fake)
                g_gradients = tape.gradient(g_error, G.trainable_variables)      
                G_solver.apply_gradients(zip(g_gradients, G.trainable_variables))

            if (iter_count % show_every == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count,d_total_error,g_error))
                imgs_numpy = fake_images.cpu().numpy()
                images.append(imgs_numpy[0:16])
                
            iter_count += 1
    
    # random noise fed into our generator
    z = sample_noise(batch_size, noise_size)
    # generated images
    G_sample = G(z)
    
    return images, G_sample[:16]

class MNIST(object):
    def __init__(self, batch_size, shuffle=False):
        """
        Construct an iterator object over the MNIST data
        
        Inputs:
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        train, _ = tf.keras.datasets.mnist.load_data()
        X, y = train
        X = X.astype(np.float32)/255
        X = X.reshape((X.shape[0], -1))
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B)) 

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.shape) for p in model.weights])
    return param_count