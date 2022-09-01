from numpy import *
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import multiply


def rigid_transform_2D(A, B):
    assert len(A) == len(B)
    N = A.shape[0];
    mu_A = mean(A, axis=0)
    mu_B = mean(B, axis=0)

    AA = A - tile(mu_A, (N, 1))
    BB = B - tile(mu_B, (N, 1))
    H = transpose(AA) * BB

    U, S, Vt = linalg.svd(H)
    R = Vt.T * U.T

    if linalg.det(R) < 0:
        print("Reflection detected")
        Vt[1, :] *= -1
        R = Vt.T * U.T

    t = -R * mu_A.T + mu_B.T

    return R, t

if __name__=='__main__':

    R = mat(random.rand(2,2))
    t = mat(random.rand(2,1))

    U,S,Vt = linalg.svd(R)
    R = U*Vt
    if linalg.det(R) < 0:
        Vt[1,:]*=-1
        R = U*Vt

    n = 10

    A = mat(random.rand(n,2))
    B = R*A.T + tile(t,(1,n))
    B = B.T

    ret_R, ret_t = rigid_transform_2D(A,B)
    A2 = (ret_R*A.T)+ tile(ret_t,(1,n))
    A2 =A2.T

    err = A2-B

    # err = multiply(err,err)
    # err = sum(err)
    # rmse = sqrt(err/n)
    #
    print("points A2")
    print(A2)
    print("")

    print("points B")
    print(B)
    print("")

    print(err)
    # print(rmse)
    # fig = plt.figure()
    # ax=fig.add_subplot(111,projection='3d')
    # ax.scatter(A[:,0],A[:,1],A[:,2])
    # ax.scatter(B[:,0],B[:,1],B[:,2],s=100,marker='x')
    # ax.scatter(A2[:,0],A2[:,1],A2[:,2],s=100,marker= 'o')
    # plt.legend()
    # plt.show()
