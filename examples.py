import numpy as np
import scipy.fft as spfft
import matplotlib.pyplot as plt

image_name = "alex.jpg"

# ================================================================================================
#   _____   _____ _______   ======================================================================
#  |  __ \ / ____|__   __|  ======================================================================
#  | |  | | |       | |     ======================================================================
#  | |  | | |       | |     ======================================================================
#  | |__| | |____   | |     ======================================================================
#  |_____/ \_____|  |_|     ======================================================================
#                           ======================================================================
#                           ======================================================================
# ================================================================================================

image      = plt.imread(image_name)
image_type = image_name[-3:]
image_name = image_name[:-4]
image      = np.array(image, dtype = float)/255.0
plt.figure()
plt.imshow(image)
plt.xlabel("Pixel location")
plt.ylabel("Pixel location")
plt.title("Original image")
plt.draw()

image_dct = spfft.dctn(image, axes = (0, 1))
image_dct /= np.max(image_dct)/2e2
image_dct = np.clip(image_dct, -1, 1)
plt.figure()
plt.imshow(np.abs(image_dct))
plt.xlabel("Spatial frequency")
plt.ylabel("Spatial frequency")
plt.title("DCT of image")
plt.imsave(f"output/{image_name}_dct.{image_type}", np.abs(image_dct))
plt.draw()


# ================================================================================================
#   _____  _____ _______         =================================================================
#  |_   _|/ ____|__   __|/\      =================================================================
#    | | | (___    | |  /  \     =================================================================
#    | |  \___ \   | | / /\ \    =================================================================
#   _| |_ ____) |  | |/ ____ \   =================================================================
#  |_____|_____/   |_/_/    \_\  =================================================================
#                                =================================================================
#                                =================================================================
# ================================================================================================

# Meta-parameters --------------------------------------------------------------------------------
step                 = 1e-1 # Gradient method step size
regularisation       = 3e2  # Weighting between data fitting and sparsity in loss
number_of_iterations = 1e2

# Subsample image --------------------------------------------------------------------------------
# Randomly select ~20% of pixels to sample
sample_set                       = np.random.randint(10, size = image_dct.shape[:-1]) >= 8
not_sample_set                   = np.logical_not(sample_set) # The complement of this set
image_subsampled                 = image.copy()
image_subsampled[not_sample_set] = np.array([0.0, 1.0, 0.0])  # Set "unsampled" pixels to green
plt.figure()
plt.imshow(image_subsampled)
plt.imsave(f"output/{image_name}_subsampled.{image_type}", image_subsampled)
plt.draw()

# ISTA ------------------------------------------------------------------------------------------
image_subsampled[not_sample_set] = 0.0                                    # Remove unsampled pixels from transform         
image_guess                      = image_subsampled.copy()
image_guess_dct                  = spfft.dctn(image_guess, axes = (0, 1)) # Use this to initialise guess of image

for iteration in range(int(number_of_iterations)):
  image_guess                 = spfft.idctn(image_guess_dct, axes = (0, 1)) # Find image from current dct guess
  image_guess[not_sample_set] = 0                                           # Only consider sampled pixels
  difference                  = image_guess - image_subsampled              # See how much it deviates from measurements
  difference_dct              = spfft.dctn(difference, axes = (0, 1))       # Transform back to frequency domain
  image_guess_dct            -= 2*step*difference_dct                       # Update dct guess to better fit data
  # Update dct guess to be more sparse
  # This operation is called "soft thresholding"
  image_guess_dct             = np.sign(image_guess_dct)*np.maximum(np.abs(image_guess_dct) - step*regularisation, 0)
  print(f"Iteration: {iteration} of {int(number_of_iterations)}", end = "\r")
print(f"Iteration: {int(number_of_iterations)} of {int(number_of_iterations)}\nDone!")

# Show result -----------------------------------------------------------------------------------
image_guess = spfft.idctn(image_guess_dct, axes = (0, 1))
image_guess = np.clip(image_guess, 0, 1)
plt.figure()
plt.imshow(image_guess)
plt.imsave(f"output/{image_name}_ista.{image_type}", image_guess)
plt.draw()

plt.show()