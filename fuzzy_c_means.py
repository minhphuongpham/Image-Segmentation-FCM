#!/usr/bin/env python
import numpy as np
import skfuzzy as fuzz
#from geotiff.io import IO
import cv2

# load original image
DIR = 'home/scikit-image-clustering-scripts'
#img = cv2.imread('Input.png',0)
#print img
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--input', metavar='FILENAME', default = 'image.png',
                    help='input image file name')
args = vars(ap.parse_args())

# load the image and perform pyramid mean shift filtering
# to aid the thresholding step
image = cv2.imread(args["input"])
# pixel intensities
I1 = image.reshape((-1, 3))
I = I1.T
print I
# params
n_centers = 10
fuzziness_degree = 2
error = 0.005
maxiter = 1000

# fuzz c-means clustering
centers, u, u0, d, jm, n_iters, fpc = fuzz.cluster.cmeans(
    I,
    c=n_centers,
    m=fuzziness_degree,
    error=error,
    maxiter=maxiter,
    init=None
)

img_clustered = np.argmax(u, axis=0).astype(int)
#print img_clustered
# display clusters
print img_clustered
print centers

a = []
b = []
for i in range(0, n_centers):
    a.append(np.sum(abs(centers[i] - [33,33,40])))
    b.append(np.sum(abs(centers[i] - [140,140, 56])))
a1 = np.argmin(a)
print a
print a1

b1 = np.argmin(b)
print b
print(b1)
img_clustered2 = img_clustered.copy()
[len(centers)+1 if (x !=a1 and x != b1) else x for x in img_clustered2]
img_clustered2= [100 if x == a1 else x for x in img_clustered2]
img_clustered2= [101 if x == b1 else x for x in img_clustered2]
img_clustered2 = [2 if (x != 100 and x != 101) else x for x in img_clustered2]
img_clustered2= [0 if x == 100 else x for x in img_clustered2]
img_clustered2= [1 if x == 101 else x for x in img_clustered2]

#print(img_clustered2)
center2 = np.uint8([[0,0,0],[255,0,0],[255,255,255], [255,255,0],[0,0,255],[0,255,255],[255,0,255]])
#new = center2[img_clustered]
#print new
center3 = np.uint8([[0,0,0],[255,0,0],[255,255,255]])
#new2 = new.reshape((image.shape))
#cv2.imwrite('output6.png', new2)
cv2.imwrite('output8.png', center3[img_clustered2].reshape((image.shape)))
def daviesbouldin(X, labels, centroids):

    import numpy as np
    from scipy.spatial.distance import pdist, euclidean

    nbre_of_clusters = len(centroids) #Get the number of clusters
    distances = [[] for e in range(nbre_of_clusters)] #Store intra-cluster distances by cluster
    distances_means = [] #Store the mean of these distances
    DB_indexes = [] #Store Davies_Boulin index of each pair of cluster
    second_cluster_idx = [] #Store index of the second cluster of each pair
    first_cluster_idx = 0 #Set index of first cluster of each pair to 0

    # Step 1: Compute euclidean distances between each point of a cluster to their centroid
    for cluster in range(nbre_of_clusters):
        for point in range(X[labels == cluster].shape[0]):
            distances[cluster].append(euclidean(X[labels == cluster][point], centroids[cluster]))

    # Step 2: Compute the mean of these distances
    for e in distances:
        distances_means.append(np.mean(e))

    # Step 3: Compute euclidean distances between each pair of centroid
    ctrds_distance = pdist(centroids)

    # Tricky step 4: Compute Davies-Bouldin index of each pair of cluster
    for i, e in enumerate(e for start in range(1, nbre_of_clusters) for e in range(start, nbre_of_clusters)):
        second_cluster_idx.append(e)
        if second_cluster_idx[i-1] == nbre_of_clusters - 1:
            first_cluster_idx += 1
        DB_indexes.append((distances_means[first_cluster_idx] + distances_means[e]) / ctrds_distance[i])

    # Step 5: Compute the mean of all DB_indexes
    print("DAVIES-BOULDIN Index: %.5f" % np.mean(DB_indexes))

#daviesbouldin(I1, img_clustered, centers)

import numpy as np
import scipy.spatial


def pairwise_squared_distances(A, B):
    return scipy.spatial.distance.cdist(A, B) ** 2


def calculate_covariances(x, u, v, m):
    c, n = u.shape
    d = v.shape[1]

    um = u ** m

    covariances = np.zeros((c, d, d))

    for i in range(c):
        xv = x - v[i]
        uxv = um[i, :, np.newaxis] * xv
        covariances[i] = np.einsum('ni,nj->ij', uxv, xv) / np.sum(um[i])

    return covariances


def pc(x, u, v, m):
    c, n = u.shape
    return np.square(u).sum() / n


def npc(x, u, v, m):
    n, c = u.shape
    return 1 - c / (c - 1) * (1 - pc(x, u, v, m))


def fhv(x, u, v, m):
    covariances = calculate_covariances(x, u, v, m)
    return sum(np.sqrt(np.linalg.det(cov)) for cov in covariances)


def fs(x, u, v, m):
    n = x.shape[0]
    c = v.shape[0]

    um = u ** m

    v_mean = v.mean(axis=0)

    d2 = pairwise_squared_distances(x, v)

    distance_v_mean_squared = np.linalg.norm(v - v_mean, axis=1, keepdims=True) ** 2

    return np.sum(um.T * d2) - np.sum(um * distance_v_mean_squared)


def xb(x, u, v, m):
    n = x.shape[0]
    c = v.shape[0]

    um = u ** m

    d2 = pairwise_squared_distances(x, v)
    v2 = pairwise_squared_distances(v, v)

    v2[v2 == 0.0] = np.inf

    return np.sum(um.T * d2) / (n * np.min(v2))


def bh(x, u, v, m):
    n, d = x.shape
    c = v.shape[0]

    d2 = pairwise_squared_distances(x, v)
    v2 = pairwise_squared_distances(v, v)

    v2[v2 == 0.0] = np.inf

    V = np.sum(u * d2.T, axis=1) / np.sum(u, axis=1)

    return np.sum(u ** m * d2.T) / n * 0.5 * np.sum(np.outer(V, V) / v2)


def bws(x, u, v, m):
    n, d = x.shape
    c = v.shape[0]

    x_mean = x.mean(axis=0)

    covariances = calculate_covariances(x, u, v, m)

    sep = np.einsum("ik,ij->", u ** m, np.square(v - x_mean))
    comp = sum(np.trace(covariance) for covariance in covariances)

    return sep / comp


methods = [pc, npc, fhv, fs, xb, bh, bws]
targets = "max max min min min min max".split()
#print fpc
#print pc(I1, u, centers, 2)
#print npc(I1, u, centers, 2)
#print fhv(I1, u, centers, 2)
#print fs(I1, u, centers, 2)
#print xb(I1, u, centers, 2)
#print bh(I1, u, centers, 2)
#print bws(I1, u, centers, 2)

