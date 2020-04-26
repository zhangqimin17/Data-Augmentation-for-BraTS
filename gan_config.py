# Root directory for dataset
dataroot = "./data"

# Number of workers for dataloader
#workers = 5

# Batch size during training
batch_size = 6

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 240

# Number of channels in the training images. For BraTS images this is 155
nc = 155

# Size of z latent vector (i.e. size of generator input)
nz = 155

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 128

# Number of training epochs
num_epochs = 70

# Learning rate for optimizers
lr = 0.002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.9
beta2 = 0.999

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1