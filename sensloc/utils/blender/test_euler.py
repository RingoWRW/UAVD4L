import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

def qvec2rotmat(qvec):  #!wxyz
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def qvec2euler(ouput_path, input_path):

    with open(ouput_path,'w') as file_w:
        with open(input_path, 'r') as fg:
            for line in fg:
                data_line = line.strip("\n").split(' ')
                name = data_line[0]
                q, t = np.split(np.array(data_line[1:], float), [4])
                Rotate = np.asmatrix(qvec2rotmat(q)).transpose()  #c2w
                T = np.identity(4)
                T[0:3,0:3] = Rotate
                T[0:3,3] = -Rotate.dot(t)
                R1 = np.asmatrix(qvec2rotmat(q))
                ret = R.from_matrix(R1)
                euler = ret.as_euler('zxy', degrees=True)
            
                # tvec = -Rotate.dot(t)
                # print(type(tvec), tvec.shape)
                # outline = name + ' ' +str(T[0][3]) + ' ' +str(T[1][3]) + ' ' +str(T[2][3]) + '\n'
                outline = name + ' ' + str(euler[0]) + ' ' + str(euler[1]) + ' ' + str(euler[2]) + ''+str(T[0][3]) + ' ' +str(T[1][3]) + ' ' +str(T[2][3]) +'\n'
                file_w.write(outline)

def main():

    parser = argparse.ArgumentParser(description="qvec2euler.")
    parser.add_argument("--ouput_path", default="/home/ubuntu/Documents/code/SensLoc/outputs/targetloc/CityofStars_query1/WideAngle_hloc_loftr_openibl_euler.txt",)
    parser.add_argument("--input_path", default="/home/ubuntu/Documents/code/SensLoc/outputs/targetloc/CityofStars_query1/WideAngle_hloc_loftr_openibl_1.txt")
    args = parser.parse_args()
    print("----transform success----")
    qvec2euler(args.ouput_path, args.input_path)
    print("----transform success----")

if __name__ == "__main__":

    main()


