import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#print(os.listdir('maps'))
#print(os.listdir('maps/train')[:5])
#print(os.listdir('maps/val')[:5])
#print("Train images:", len(os.listdir('maps/train')))
#print("Validation images:", len(os.listdir('maps/val')))

def list_jpgs(path):
    return [f for f in os.listdir(path) if f.lower().endswith('.jpg')]

print("Dataset folders:", os.listdir('maps'))
print("Train sample files:", list_jpgs('maps/train')[:5])
print("Val sample files:", list_jpgs('maps/val')[:5])
print("Train images:", len(list_jpgs('maps/train')))
print("Validation images:", len(list_jpgs('maps/val')))

TRAIN_PATH = 'maps/train/*.jpg'
VAL_PATH   = 'maps/val/*.jpg'
DATASET_PATH = 'maps'

print("Dataset folders:", os.listdir(DATASET_PATH))
print("Train images count:", len(os.listdir(os.path.join(DATASET_PATH, 'train'))))
print("Validation images count:", len(os.listdir(os.path.join(DATASET_PATH, 'val'))))
sample_image_path = os.path.join(DATASET_PATH, 'train', os.listdir(os.path.join(DATASET_PATH, 'train'))[0])

sample_image = tf.io.read_file(sample_image_path)
sample_image = tf.image.decode_jpeg(sample_image)

plt.figure(figsize=(6, 3))
plt.imshow(sample_image)
plt.axis('off')
plt.title('Raw Map | Satellite Paired Image')
plt.show(block = False)
plt.pause(2)
plt.close()

# Global configuration parameters

BUFFER_SIZE = 400
BATCH_SIZE = 1        # pix2pix standard
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Set random seeds for reproducibility

tf.random.set_seed(42)
np.random.seed(42)

def load_image(image_file):
    # Read image from disk
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)

    # Split image into left (map) and right (satellite)
    w = tf.shape(image)[1]
    w_half = w // 2

    input_image = image[:, :w_half, :]   # Map
    target_image = image[:, w_half:, :]  # Satellite

    return input_image, target_image

def resize(input_image, target_image, height=256, width=256):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_image = tf.image.resize(target_image, [height, width],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, target_image

def normalize(input_image, target_image):
    input_image = (input_image / 127.5) - 1
    target_image = (target_image / 127.5) - 1
    return input_image, target_image

def load_image_train(image_file):
    input_image, target_image = load_image(image_file)
    input_image, target_image = resize(input_image, target_image)
    input_image, target_image = normalize(input_image, target_image)
    return input_image, target_image

# Training dataset
train_dataset = tf.data.Dataset.list_files('maps/train/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# Validation dataset
val_dataset = tf.data.Dataset.list_files('maps/val/*.jpg')
val_dataset = val_dataset.map(load_image_train,
                              num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)

train_dataset = train_dataset.take(50)   # only 50 steps
val_dataset   = val_dataset.take(5)      # only 5 validation samples


for input_image, target_image in train_dataset.take(1):
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.title("Input Map")
    plt.imshow((input_image[0] + 1) / 2)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Target Satellite")
    plt.imshow((target_image[0] + 1) / 2)
    plt.axis('off')

    plt.show(block = False)
    plt.pause(2)
    plt.close()

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters, size, strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters, size, strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    # Encoder (Downsampling)
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (128x128)
        downsample(128, 4),                         # (64x64)
        downsample(256, 4),                         # (32x32)
        downsample(512, 4),                         # (16x16)
        downsample(512, 4),                         # (8x8)
        downsample(512, 4),                         # (4x4)
        downsample(512, 4),                         # (2x2)
        downsample(512, 4),                         # (1x1)
    ]

    # Decoder (Upsampling)
    up_stack = [
        upsample(512, 4, apply_dropout=True),       # (2x2)
        upsample(512, 4, apply_dropout=True),       # (4x4)
        upsample(512, 4, apply_dropout=True),       # (8x8)
        upsample(512, 4),                           # (16x16)
        upsample(256, 4),                           # (32x32)
        upsample(128, 4),                           # (64x64)
        upsample(64, 4),                            # (128x128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        3, 4, strides=2, padding='same',
        kernel_initializer=initializer,
        activation='tanh'
    )  # (256x256x3)

    x = inputs
    skips = []

    # Downsampling pass
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling pass + skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
generator.summary()

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    # Inputs
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    # Concatenate input and target
    x = tf.keras.layers.Concatenate()([inp, tar])  # (256,256,6)

    # Downsampling layers
    down1 = downsample(64, 4, False)(x)   # (128x128)
    down2 = downsample(128, 4)(down1)    # (64x64)
    down3 = downsample(256, 4)(down2)    # (32x32)

    # Extra convolution layers
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1,
        kernel_initializer=initializer,
        use_bias=False
    )(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    # Output layer (PatchGAN output)
    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer
    )(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()
discriminator.summary()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output),
                                 disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output),
                           disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    LAMBDA = 50
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=2e-4, beta_1=0.5
)

discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=2e-4, beta_1=0.5
)

@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        # Generator forward pass
        gen_output = generator(input_image, training=True)

        # Discriminator forward pass
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # Compute losses
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )

        disc_loss = discriminator_loss(
            disc_real_output, disc_generated_output
        )
        
                # Calculate gradients
    generator_gradients = gen_tape.gradient(
            gen_total_loss, generator.trainable_variables
        )
    discriminator_gradients = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        # Apply gradients
    generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )
    discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )
        
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss
    
def generate_images(model, test_input, target):
    prediction = model(test_input, training=True)

    plt.figure(figsize=(12,4))

    display_list = [
        test_input[0],
        target[0],
        prediction[0]
    ]
    title = ['Input Map', 'Ground Truth Satellite', 'Predicted Satellite']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow((display_list[i] + 1) / 2)
        plt.axis('off')

    plt.show(block = False)
    plt.pause(2)
    plt.close()


def train(train_dataset, val_dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        print(f"\nEpoch {epoch+1}/{epochs}")

        for n, (input_image, target) in enumerate(train_dataset):
            gen_loss, gan_loss, l1_loss, disc_loss = train_step(
                input_image, target
            )
            
            print(
                f"Epoch {epoch+1} | "
                f"Step {n:04d} | "
                f"Gen Total: {gen_loss:.4f} | "
                f"GAN: {gan_loss:.4f} | "
                f"L1: {l1_loss:.4f} | "
                f"Disc: {disc_loss:.4f}"
                )

        print(f"Epoch {epoch+1} finished in {time.time() - start:.2f} seconds")

            


        # Visualize results on validation data
        for example_input, example_target in val_dataset.take(1):
            generate_images(generator, example_input, example_target)

        print(f"Time taken for epoch {epoch+1}: {time.time() - start:.2f} sec")

EPOCHS = 70   # You can reduce to 5–10 for quick runs

train(train_dataset, val_dataset, EPOCHS)






#if n % 100 == 0:
              #  print(f"Step {n}: "
               #       f"Gen Loss={gen_loss:.4f}, "
                #      f"Disc Loss={disc_loss:.4f}")





