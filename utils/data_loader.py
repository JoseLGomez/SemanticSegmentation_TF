import numpy as np
import skimage.io as io
from skimage.color import rgb2gray, gray2rgb
import skimage.transform
import ntpath
import os

# List the subdirectories in a directory
def list_subdirs(directory):
    subdirs = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            subdirs.append(subdir)
    return subdirs

# Get file names
def Get_filenames(directory):
    subdirs = list_subdirs(directory)
    subdirs.append(directory)
    file_names = []
    file_paths = []
    for subdir in subdirs:
        subpath = os.path.join(directory, subdir)
        for fname in os.listdir(subpath):
            if has_valid_extension(fname):
                file_paths.append(os.path.join(directory, subdir, fname))
                file_names.append(fname)
    return file_paths, file_names

# Load image
def Load_image(image_path, resize, grayscale, order = 1):
    img = io.imread(image_path)
    # Resize
    if resize is not None:
        img = skimage.transform.resize(img, resize, order=order, preserve_range=True, mode='constant')
    # Color conversion
    if len(img.shape) == 2 and not grayscale:
        img = gray2rgb(img)
    elif len(img.shape) > 2 and img.shape[2] == 3 and grayscale:
        img = rgb2gray(img)
    return img

# Checks if a file is an image
def has_valid_extension(fname, white_list_formats={'png', 'jpg', 'jpeg',
                        'bmp', 'tif'}):
    for extension in white_list_formats:
        if fname.lower().endswith('.' + extension):
            return True
    return False

class Data_loader():
    def __init__(self, cf, image_path, num_images, resize, gt_path=None):
        self.cf = cf
        self.images = []
        self.offset = 0
        self.img_path = image_path
        self.gt_path = gt_path
        self.num_images = num_images
        self.num_batches = None
        self.resize = resize
    
    def Load_dataset(self, batch_size):
        _ , self.images = Get_filenames(self.img_path)
        #annotations = Get_filenames(gt_path)
        if len(self.images) < self.num_images or self.num_images == -1:
            self.num_images = len(self.images)
        else:
            self.num_images = self.num_images
        self.num_batches = self.num_images/batch_size  
        self.indexes = np.arange(len(self.images))

    def Shuffle(self):
        np.random.shuffle(self.indexes)
        
    def Next_batch(self, batch_size, crop=False):
        if batch_size > 1:
            if crop:
                batch_x = np.zeros((batch_size,self.cf.crop_train[0],self.cf.crop_train[1],
                                    self.cf.image_channels))
                batch_y = np.zeros((batch_size,self.cf.crop_train[0],self.cf.crop_train[1],1))
            else:
                batch_x = np.zeros((batch_size,self.resize[0],self.resize[1],
                                    self.cf.image_channels))
                batch_y = np.zeros((batch_size,self.resize[0],self.resize[1],1))
        # Build batch of image data
        for i in range(batch_size):
            # Load image
            fname = self.images[self.indexes[self.offset + i]]
            img = Load_image(os.path.join(self.img_path, fname), self.resize, 
                                self.cf.grayscale, order=1)
            x = np.array(img)
            #assert not np.any(np.isnan(x))
            # Load GT image
            gt_img = Load_image(os.path.join(self.gt_path, fname), self.resize, 
                                grayscale=True, order=0)
            y = np.array(gt_img)
            y = np.expand_dims(y, axis=2)
            if crop:
                x, y = Preprocess_IO().ApplyCrop(x, y, self.cf)  
            x = Preprocess_IO().Preproces_input(x, self.cf)  
            #assert not np.any(np.isnan(y))
            if batch_size > 1:
                batch_x[i] = x
                batch_y[i] = y
            else:
                batch_x = np.expand_dims(x, axis=0)
                batch_y = np.expand_dims(y, axis=0)
        self.offset += batch_size
        return batch_x, batch_y

    def Next_batch_pred(self, batch_size):
        batch_names = []
        if batch_size > 1:
            batch_x = np.zeros((batch_size,self.resize[0],self.resize[1],
                                self.cf.image_channels))
        # Build batch of image data
        for i in range(batch_size):
            # Load image
            fname = self.images[self.indexes[self.offset + i]]
            img = Load_image(os.path.join(self.img_path, fname), self.resize, 
                                self.cf.grayscale, order=1)
            x = np.array(img)
            x = Preprocess_IO().Preproces_input(x, self.cf)

            if batch_size > 1:
                batch_x[i] = x
            else:
                batch_x = np.expand_dims(x, axis=0)
            batch_names.append(fname) 
        self.offset += batch_size
        return batch_x, batch_names

    def Reset_Offset(self):
        self.offset = 0
            
class Preprocess_IO():
    def __init__(self):
        pass
    
    def Rescale(self, image,rescale):
        return image * rescale
        
    def Mean_norm(self, image, mean):
        return image - mean
        
    def Std_norm(self, image, std):
        return image/(std + 1e-7)
        
    def Preproces_input(self, image, cf):
        if cf.rescale is not None:
            image = self.Rescale(image,cf.rescale)
        if cf.mean is not None:
            cf.mean = np.asarray(cf.mean, dtype=np.float32)
            image = self.Mean_norm(image, cf.mean)
        if cf.std is not None:
            cf.std = np.asarray(cf.std, dtype=np.float32)
            image = self.Std_norm(image, cf.std)
        return image

    def ApplyCrop(self, x, y, cf):
        if cf.crop_train[0] < cf.size_image_train[0]:
            top = np.random.randint(cf.size_image_train[0] - cf.crop_train[0])
        if cf.crop_train[1] < cf.size_image_train[1]:
            left = np.random.randint(cf.size_image_train[1] - cf.crop_train[1])

        x = x[..., top:top+cf.crop_train[0], left:left+cf.crop_train[1], :]
        y = y[..., top:top+cf.crop_train[0], left:left+cf.crop_train[1], :]
        return x, y
