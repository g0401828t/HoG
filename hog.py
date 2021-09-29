## for simple visualization
"""
#importing required libraries
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt

#reading the image
img = imread('pedestrians.jpg')
plt.axis("off")
plt.imshow(img)
print(img.shape)

# resizing image
resized_img = resize(img, (128*4, 64*4))
# plt.axis("off")
# plt.imshow(resized_img)
# plt.show()
print(resized_img.shape)

## creating hog features
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=True)
print(fd.shape)
print(hog_image.shape)
# plt.axis("off")
# plt.imshow(hog_image, cmap="gray")
# plt.show()

# save the images
plt.imsave("resized_img.jpg", resized_img)
plt.imsave("hog_image.jpg", hog_image, cmap="gray")
"""




## implemented in scratch
import os, sys
import matplotlib.pyplot as plt
import matplotlib.image as iread
from PIL import Image
import numpy as np

cwd = os.getcwd()

cell = [8, 8]  # kernel = (8, 8)
incr = [8, 8]  # stride = 8
bin_num = 9    # histogram_bin_num
im_size = [64,128]  # img_size for resizing image


# image path must be wrt current working directory
def create_array(image_path):
	
    image = Image.open(os.path.join(cwd,image_path)).convert('L')
    image_array = np.asarray(image,dtype=float)
	
    return image_array


"""Stage 1"""
## uses a [-1 0 1 kernel] to calculate gradients.
## calculate grad(angle of gradient) and mag(magnitude of gradient) for every pixels.
## for the angle, "unsigned" angle is better for the algorithm, so add 180 for negative angles.
def create_grad_array(image_array):
    image_array = Image.fromarray(image_array)
    if not image_array.size == im_size:
        image_array = image_array.resize(im_size, resample=Image.BICUBIC)
	
    image_array = np.asarray(image_array,dtype=float)
	
    # gamma correction
    # image_array = (image_array)**2.5

    # local contrast normalization
    image_array = (image_array-np.mean(image_array))/np.std(image_array)
    max_h, max_w = im_size[1],im_size[0]

    # calculation of gradients for each pixels.
    grad = np.zeros([max_h, max_w])
    mag = np.zeros([max_h, max_w])
    for h,row in enumerate(image_array):
        for w, val in enumerate(row):
            if h-1>=0 and w-1>=0 and h+1<max_h and w+1<max_w:
                dy = image_array[h+1][w]-image_array[h-1][w]
                dx = row[w+1]-row[w-1]+0.0001
                grad[h][w] = np.arctan(dy/dx)*(180/np.pi)
                if grad[h][w]<0:  # change gradients to unsigned gradients
                    grad[h][w] += 180
                mag[h][w] = np.sqrt(dy*dy+dx*dx)
	
    return grad,mag

def write_hog_file(filename, final_array):
    print('Saving '+filename+' ........\n')
    np.savetxt(filename,final_array)

def read_hog_file(filename):
    return np.loadtxt(filename)


# calcualte histogram for 0, 20, .., 180.
# num_bins = 9
def calculate_histogram(array,weights):
    bins_range = (0, 180)
    bins = bin_num
    hist,_ = np.histogram(array,bins=bins,range=bins_range,weights=weights)
    return hist

"""Stage 2"""
## 8*8 kernel로 stride=8로 돌아가면서 histogram 생성
## (128, 64)의 경우 총 128개의 histogram 생성됨
def create_hog_features(grad_array,mag_array):
    # kernel size(cell) = 8, stride(incr) = 8 일때 iteration +> 총 16*8=128개의 histogram
    max_h = int(((grad_array.shape[0]-cell[0])/incr[0])+1)  # 15+1
    max_w = int(((grad_array.shape[1]-cell[1])/incr[1])+1)  # 7+1
    cell_array = []
    w = 0
    h = 0
    i = 0
    j = 0

	#Creating 8X8 cells
    while i<max_h:
        w = 0
        j = 0

        while j<max_w:
            for_hist = grad_array[h:h+cell[0],w:w+cell[1]]
            for_wght = mag_array[h:h+cell[0],w:w+cell[1]]
			
            val = calculate_histogram(for_hist,for_wght)
            cell_array.append(val)
            j += 1
            w += incr[1]  # stride(incr) = 8 만큼 cell을 이동한다

        i += 1
        h += incr[0]  # stride(incr) = 8 만큼 cell을 이동한다
    
    # cell_array: (128, 9)
    cell_array = np.reshape(cell_array,(max_h, max_w, bin_num))
    # cell_array: (16, 8, 9)



    # Concatenate and Normalising blocks of cells
    # 2*2 kernel, stride: 1, concate and normalize
    # (16, 8, 9) ->  (15, 7, 9*4)
    # 4 histograms are concatenated and normalized for every 2x2 kernel with stride 1
    block = [2,2]
    max_h = int((max_h-block[0])+1)
    max_w = int((max_w-block[1])+1)
    block_list = []
    w = 0
    h = 0
    i = 0
    j = 0

    while i<max_h:
        w = 0
        j = 0

        while j<max_w:
            for_norm = cell_array[h:h+block[0],w:w+block[1]]
            mag = np.linalg.norm(for_norm)
            """Stage 3""" ## after normalizing, flatten and concatenate
            arr_list = (for_norm/mag).flatten().tolist()  
            block_list += arr_list
            j += 1
            w += 1

        i += 1
        h += 1

    # (15, 7, 9x4) -> flatten: 3780
    return block_list

#image_array must be an array
#returns a 288 features vector from image array
def apply_hog(image_array):
    gradient,magnitude = create_grad_array(image_array)
    hog_features = create_hog_features(gradient,magnitude)
    hog_features = np.asarray(hog_features,dtype=float)
    hog_features = np.expand_dims(hog_features,axis=0)

    return hog_features

#path must be image path
#returns final features array from image_path
def hog_from_path(image_path):
    image_array = create_array(image_path)
    final_array = apply_hog(image_array)
	
    return final_array

#Creates hog files
def create_hog_file(image_path,save_path):
    image_array = create_array(image_path)
    final_array = apply_hog(image_array)
    write_hog_file(save_path,final_array)

if __name__ == '__main__':
    create_hog_file('pedestrians.jpg','pedestrians.txt')
    mg = read_hog_file('pedestrians.txt')
    print(mg)
    print(mg.shape)