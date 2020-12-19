
import numpy as np
np.random.seed(1) # random seed for consistency, debugging same results every time


#We'll first create 2 classes

#each with 3 features
# create class 1 random sampled 3 X 20 data set


#Draw random samples from a multivariate normal distribution.
#also called Gaussian distribution
#https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Gaussian_2d.png/786px-Gaussian_2d.png
#uses mean to define center ,covariance for width, or standard deviation. how wide. how spread.
#btw Covariance is a measure of how changes in one variable are associated with changes in a second variable. 
#Specifically, covariance measures the degree to which two variables are linearly associated.
mu_vec1 = np.array([0,0,0])  # sample mean
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) # sample covariance
#Transpose of a Matrix. A matrix which is 
#formed by turning all the rows of a given matrix into columns and vice-versa. 
#convenience, for printing
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
print(class1_sample)
class1_sample.shape


# create class 2 random sampled 3 x 20 data set


mu_vec2 = np.array([1,1,1]) # sample mean
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #sample covariance
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
print(class2_sample)


get_ipython().run_line_magic("pylab", " inline")

#plotting
from matplotlib import pyplot as plt

#figure, width and height
fig = plt.figure(figsize=(8,8))
#3d subplot
#These are subplot grid parameters encoded as a single integer, so 1x1 grid 1st subplot
ax = fig.add_subplot(111, projection='3d')
#fontsize
plt.rcParams['legend.fontsize'] = 10

#plot samples
ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:],
        'o', markersize=12, color='blue', label='class1')
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:],
        '^', markersize=12,  color='green', label='class2')

ax.legend(loc='upper right')

plt.show()


#make it one big dataset
#3 x 40 still 3 features

all_samples = np.concatenate((class1_sample, class2_sample), axis=1)

all_samples
all_samples.shape


all_samples.T


# step 2. compute the d dimensional mean vector, to help compute covariance matrix


#mean for each feature
mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

#3D mean vector
mean_vector = np.array([[mean_x],[mean_y],[mean_z]])
print('Mean Vector:\n', mean_vector)


## Compute the Covariance

#covariane matrix models relationship between our variables. the variance between each
#http://support.minitab.com/en-us/minitab/17/topic-library/modeling-statistics/anova/anova-statistics/what-is-the-variance-covariance-matrix/
#http://stats.seandolinar.com/making-a-covariance-matrix-in-r/
#Variance is the degree by which a random vairable changes with respect to its expected value
#Covariance is the degree by which two different random variables change with respect to each other. 
#measures relationship between each feature

cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])

print('Covariance Matrix:\n', cov_mat)


# Step 4. computer eigenvectors and eigenvalues

#Eigenvalues/vectors are instrumental to understanding electrical circuits, mechanical systems, ecology and 
#even Google's PageRank algorithm. 
#Eigenvectors make understanding linear transformations easy.
#They are the "axes" (directions) along which a linear transformation acts simply by 
#"stretching/compressing" and/or "flipping"; 
#eigenvalues give you the factors by which this compression occurs.
#There are a lot of problems that can be modeled with linear transformations, and 
#the eigenvectors give very simply solution
#The more directions you have along which you understand the behavior of a 
#linear transformation, the easier it is to understand the linear transformation; 
#so you want to have as many linearly independent eigenvectors as possible associated 
#to a single linear transformation.
#interactive tool - http://setosa.io/ev/eigenvectors-and-eigenvalues/


# eigenvectors and eigenvalues for the from the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(cov_mat)

for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
    print('Eigenvector {}: \n\n{}'.format(i+1, eigvec_sc))
    print('Eigenvalue {} from scatter matrix: \n\n{}'.format(i+1, eig_val_sc[i]))


# step 5. sort eigenvector by decreasing value

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i])
             for i in range(len(eig_val_sc))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
    print(i[0])


# step 5.2 choose k eigenvectos w largest eigenvalues to form d x k matrix
# k is 2 

matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1),
                      eig_pairs[1][1].reshape(3,1)))
print('Matrix W:\n', matrix_w)


# step 6. use d x k to transform samples to new subspace

#dot product between orignal matrix and eigen pairs

transformed = matrix_w.T.dot(all_samples) 
assert transformed.shape == (2,40), "The matrix is not 2x40 dimensional."
# assert verifica si el tamaño de la matriz transformed es correcta


plt.figure(figsize=(8,8))
plt.plot(transformed[0,0:20], transformed[1,0:20],
         'o', markersize=12, color='green', label='class1')
plt.plot(transformed[0,20:40], transformed[1,20:40],
         '^', markersize=12, color='red', label='class2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')

plt.show()



