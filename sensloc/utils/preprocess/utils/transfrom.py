import numpy as np

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


def rotation_to_quat (rotation_metrix):
    a = []
    # for _,v1 in rotation_metrix.items():
    for v1 in rotation_metrix:
        a.append(float(v1))
    a_np = np.array(a)
    a_np = a_np.reshape(3, 3)
    a_qvec = rotmat2qvec(a_np)# w,x,y,z
    return a_qvec, a_np

def compute_pixel_focal(sensorWidth, sensorHeight, focallength, imgWidth, imgHeight):

    # factor = np.sqrt(36*36+24*24)
    # sensorSize = np.sqrt(np.square(sensorWidth) + np.square(sensorHeight))
    # eq_length = factor * (focallength/sensorSize)
    pixelSizeW = sensorWidth/imgWidth
    pixelSizeH = sensorHeight/imgHeight
    fx = focallength / pixelSizeW
    fy = focallength / pixelSizeH
    
    return fx, fy