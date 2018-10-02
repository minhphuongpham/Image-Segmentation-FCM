'''Spatial intuitionistic Fuzzy C-means image segmentation

Modified from https://github.com/ab93/SIFCM

Changes made:

    * Vectorize computations (> 150x speed up).
    * Indentation changed to 4 spaces.
    * Create class instance using image array instead of image file name.
    * Works on images of non-square shape.
    * Expose parameters these parameters to class instance creation:
        * m: fuzziness parameter.
        * kernel_size, kernel_shape: define neighborhood relations.
        * lam: intuitionistic parameter?
    * Add comments
    * Correct a bug in the original code:
        at line 194 of original code:

            ```
	    for j in range(self.n_clusters):
            ```

        `j` should be `i`, according to their paper.


Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
Update time: 2018-04-28 17:52:27.
'''

import numpy as np
from scipy import signal
from scipy.spatial.distance import cdist
import cv2

def getGaussianElement(a, b, stda, stdb, quad=None):
    '''Create a 2d gussian weight kernel

    <a>: int, half major axis length.
    <b>: int, half minor axis length.
    <stda>, <stdb>: floats, standard deviations on x and y axis.

    <quad>: str or None. If str, choose one in the list:
            ['ne','nw','sw','se'], specifying the quadrant.
            If None, return whole disk.

    Return <ellipse>: 2d int array, gaussian weights following:

        f(x,y) = 1/2/pi/stda/stdb * exp(- x**2/2/stda**2 - y**2/2/stdb**2).

    Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
    Update time: 2016-11-15 09:07:12.
    '''

    assert a > 0 and int(a) == a, "Major half-axis 'a' needs to be an positive integer."
    assert b > 0 and int(b) == b, "Minor half-axis 'b' needs to be an positive integer."
    assert stda > 0, "Standard deviation 'stda' needs to be an positive integer."
    assert stdb > 0, "Standard deviation 'stdb' needs to be an positive integer."
    assert quad in [None, 'ne', 'nw', 'sw', 'se'], "Quadrant should be one in the list [None, 'ne','nw','sw','se']."

    ax = np.arange(-a, a + 1)
    bx = np.arange(-b, b + 1)
    x, y = np.meshgrid(ax, bx)
    ellipse = 0.5 * np.pi / stda / stdb * np.exp(-x ** 2 / 2. / stda ** 2 - y ** 2 / 2. / stdb ** 2)

    if quad is not None:
        if quad == 'ne':
            ellipse = np.where((ellipse > 0) & (x >= 0) & (y >= 0), ellipse, 0)
        elif quad == 'nw':
            ellipse = np.where((ellipse > 0) & (x <= 0) & (y >= 0), ellipse, 0)
        elif quad == 'sw':
            ellipse = np.where((ellipse > 0) & (x <= 0) & (y <= 0), ellipse, 0)
        elif quad == 'se':
            ellipse = np.where((ellipse > 0) & (x >= 0) & (y <= 0), ellipse, 0)

    # --------------------Normalize--------------------
    ellipse = ellipse / np.max(ellipse)

    return ellipse


class FCM():
    def __init__(self, image, n_clusters, m=2, kernel_size=5,
                 kernel_shape='uniform', lam=0.5, epsilon=0.05, max_iter=300):
        '''Spatial intuitionistic Fuzzy C-means clustering on image

        <image>: 2D array, grey scale image. Can be of int type or float.
        <n_clusters>: int, number of clusters/segments to create.
        <m>: float > 1, fuzziness parameter. A large <m> results in smaller
             membership values and fuzzier clusters. Commonly set to 2.
        <kernel_size>: int >= 1, size of neighborhood.
        <kernel_shape>: str, 'uniform': equally weighted kernel function in
                                        summing weights in neighborhood.
                             'gaussian': gaussian weights for summing neighborhood.
        <lam>: float > 0, intuitionistic fuzzy parameter?
        <epsilon>: float > 0, threshold to check convergence.
        <max_iter>: int, max number of iterations.

        Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
        Update time: 2018-04-28 17:50:10.
        '''

        # -------------------Check inputs-------------------
        #if np.ndim(image) != 2:
        #    raise Exception("<image> needs to be 2D (gray scale image).")
        if n_clusters <= 0 or n_clusters != int(n_clusters):
            raise Exception("<n_clusters> needs to be positive integer.")
        if m < 1:
            raise Exception("<m> needs to be >= 1.")
        if kernel_size <= 0 or kernel_size != int(kernel_size):
            raise Exception("<kernel_size> needs to be positive integer.")
        if kernel_shape not in ['uniform', 'gaussian']:
            raise Exception("<kernel_shape> needs to be one of 'uniform', 'gaussian'.")
        if lam <= 0:
            raise Exception("<lam> needs to be > 0.")
        if epsilon <= 0:
            raise Exception("<epsilon> needs to be > 0")

        self.image = image
        self.n_clusters = n_clusters
        self.m = m
        self.kernel_size = kernel_size
        self.kernel_shape = kernel_shape
        self.lam = lam
        self.epsilon = epsilon
        self.max_iter = max_iter

        self.shape = image.shape
        self.X = image.reshape((-1,3))
        self.numPixels = len(self.X)

        # -----------------Initial weights-----------------
        self.U = np.zeros((self.numPixels, self.n_clusters))
        idx = np.arange(self.numPixels)
        for ii in range(self.n_clusters):
            idxii = idx % self.n_clusters == ii
            self.U[idxii, ii] = 1

        # -----------------Initial centers-----------------
        self.C = [image[i] for i in range(n_clusters)]
        #self.C = self.C.reshape.T

        # ---------------Neighbourhood kernel---------------
        if kernel_shape == 'uniform':
            self.kernel = np.ones([kernel_size, kernel_size, 1])
        # if kernel_shape == 'gaussian':
        #    radius = max(1,int(kernel_size/.2))
        #    stdr = max(1,int(radius/.2))
        #    self.kernel = getGaussianElement(radius,radius,stdr,stdr)

        # ----------------Initial hesitation----------------
        # self.hesitation = 1.-self.U-(1.-self.U)/(1.+2.*self.U)
    def _distance(data, centers):
        """
        Euclidean distance from each point to each cluster center.

    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.

    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.

    See Also
    --------
    scipy.spatial.distance.cdist
    """
        return cdist(data, centers).T

    def update_U(self):
        '''Compute weights'''
        c = self.n_clusters

        self.U /= (np.ones((c, 1)).dot(np.atleast_2d(self.U.T.sum(axis=0)))).T
        u_old = np.fmax(self.U, np.finfo(np.float64).eps)

        um = u_old ** self.m

    # Calculate cluster centers
        data = self.X.T

        d = cdist(self.X, self.C)
        d = np.fmax(d, np.finfo(np.float64).eps)

        jm = (um * d ** 2).sum()

        u = d ** (- 2. / (self.m - 1))
        u /= (np.ones((self.n_clusters, 1)).dot(np.atleast_2d(u.T.sum(axis=0)))).T

        return u




    def update_C(self):
        '''Compute centroid of clusters'''
        num = np.dot((self.U.T) ** self.m, self.X)
        den = np.sum(self.U ** self.m, axis=0)
        a= [num[i]/den[i] for i in range(self.n_clusters) ]
        print a
        #print self.numPixels
        return a

    def calculate_h(self):
        '''Compute neighbourhood weights'''
        kernel = np.ones((5, 5, 1))
        uu = self.U.reshape((687,644,5))
        h = signal.fftconvolve(uu, kernel, mode='same')
        h = h.reshape(self.U.shape)

        return h

    def compute_intuitionistic_U(self):
        '''Compute intuitionistic U'''
        self.hesitation = 1. - self.U - (1 - self.U) / (1 + self.lam * self.U)
        #print self.U.shape, self.hesitation.shape, (self.U+self.hesitation).shape
        return self.U + self.hesitation

    def computeNew_U(self):
        '''Compute new weights after incorporating spatial and intuitionistic'''
        p = 1
        # q = 2
        q = 3
        self.h = self.calculate_h()
        num = self.U ** p * self.h ** q
        denom = num.sum(axis=1)
        denom = np.outer(denom, 1 + np.arange(self.n_clusters))

        return np.asarray([num[i] / denom[i] for i in range( len(num))])

    def computeIntraDists(self):
        '''Compute mean intra cluster distances'''

        result = self.deFuzzify()
        c_mesh, x_mesh = np.meshgrid(self.C, self.X)
        dist = abs(c_mesh - x_mesh)
        idx = np.arange(self.n_clusters)
        match = result[:, None] - idx[None, :]
        _, count = np.unique(self.result, return_counts=True)
        sigma = (dist * (match == 0)).sum(axis=0) / count

        return sigma

    def calculate_DB_score(self):
        '''Compute DB index

        DB = 1/c * \sum_{1}^{c} max_{i != j} ( (s_i + s_j) / d_{i,j} )

        where s_i is the mean distance between centroid i and its members,
        d_{i,j} is the centroid distance btween i and j
        c is the number of clusters
        '''

        sigmas = self.computeIntraDists()

        rs = sigmas[:, None] + sigmas[None, :]
        dists = abs(self.C[:, None] - self.C[None, :])
        np.fill_diagonal(dists, np.nan)
        rs = rs / dists
        ds = np.nanmax(rs, axis=0)
        DB_score = ds.sum() / self.n_clusters

        print("DB_score = %f" % DB_score)

    def calculate_D_score(self):
        '''Compute Dunn index

        D = min{ min { d_{i,j} / max {s_k} } }

        the ratio of min inter-cluster distance to max  intra-cluster distance

        where s_i is the mean distance between centroid i and its members,
        d_{i,j} is the centroid distance btween i and j
        '''

        sigmas = self.computeIntraDists()

        denom = np.max(sigmas)
        dists = abs(self.C[:, None] - self.C[None, :]) / denom
        np.fill_diagonal(dists, np.inf)
        D_score = np.min(dists)

        print("D_score = %f" % D_score)

    def calculate_scores(self):

        # --------------Partition coefficient--------------
        self.vpc = (self.U ** 2).sum() / self.numPixels
        print("VPC =  %f" % self.vpc)

        # ----------------Partition entropy----------------
        vpe = self.U * np.log(self.U)
        self.vpe = -1 * vpe.sum() / self.numPixels
        print("VPE = %f" % self.vpe)

        # ------------------Cost function------------------
        c_mesh, x_mesh = np.meshgrid(self.C, self.X)
        numer = (self.U * (x_mesh - c_mesh) ** 2).sum()
        denom = self.numPixels * (self.C[0] - self.C[1]) ** 2
        self.vxb = numer / denom
        print("VXB = %f" % self.vxb)

        # -------------DB scorea nd Dunn index-------------
        self.calculate_DB_score()
        self.calculate_D_score()

    def form_clusters(self):
        '''Iterative training'''

        d = 100
        if self.max_iter != -1:
            i = 0
            while True:
                #print self.X
                #print self.U.shape
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                #print self.U.shape
                self.U = self.compute_intuitionistic_U()
                self.U = self.computeNew_U()
                # d = sum(abs(sum(self.U_new - old_u)))
                # this way of computing d is faster
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" % (i, d))

                if d < self.epsilon or i > self.max_iter:
                    break
                i += 1
        else:
            i = 0
            while d > self.epsilon:
                self.update_C()
                old_u = np.copy(self.U)
                self.update_U()
                d = sum(abs(sum(self.U - old_u)))
                print "loop : ", int(i)
                print d
                self.segmentImage(i)
                i += 1

        self.segmentImage()

    def deFuzzify(self):
        return np.argmax(self.U, axis=1)

    def segmentImage(self):
        '''Segment image based on max weights'''

        result = self.deFuzzify()
        self.result = result.reshape((687,644)).astype('int')

        return self.result


if __name__ == '__main__':
    from skimage import data

    # img=data.imread('brainMRI.jpg',as_grey=True)
    img = data.imread('anh.png')

    cluster = FCM(img, 5, m=2, epsilon=.05, max_iter=100, kernel_shape='uniform',
                  kernel_size=9)
    cluster.form_clusters()
    #cluster.calculate_scores()
    #result = cluster.result
    img_clustered = np.argmax(cluster.U.T, axis=0).astype(int)
    print img_clustered
    a = []
    b = []
    for i in range(0, cluster.n_clusters):
        a.append(np.sum(abs(cluster.C[i] - [40, 33, 33])))
        b.append(np.sum(abs(cluster.C[i] - [80, 135, 135])))
    a1 = np.argmin(a)
    print a
    print a1

    b1 = np.argmin(b)
    print b
    print(b1)
    img_clustered2 = img_clustered.copy()
    [len(cluster.C) + 1 if (x != a1 and x != b1) else x for x in img_clustered2]
    img_clustered2 = [100 if x == a1 else x for x in img_clustered2]
    img_clustered2 = [101 if x == b1 else x for x in img_clustered2]
    img_clustered2 = [2 if (x != 100 and x != 101) else x for x in img_clustered2]
    img_clustered2 = [0 if x == 100 else x for x in img_clustered2]
    img_clustered2 = [1 if x == 101 else x for x in img_clustered2]

# print(img_clustered2)
    center2 = np.uint8([[0, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255]])
# new = center2[img_clustered]
# print new
    center3 = np.uint8([[0, 0, 0], [255, 0, 0], [255, 255, 255]])
# new2 = new.reshape((image.shape))
# cv2.imwrite('output6.png', new2)
    cv2.imwrite('output10.png', center3[img_clustered2].reshape((img.shape)))



