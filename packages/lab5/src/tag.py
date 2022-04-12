import numpy as np
import math
'''
Adapted from: https://raceon.io/localization/
'''

def rotation_matrix_euler_angles(R: np.array) -> np.array:
    # Source: https://learnopencv.com/rotation-matrix-to-euler-angles/
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

class Tag():
    def __init__(self, tag_size, family):
        self.family = family
        self.size = tag_size
        self.locations = {}
        self.orientations = {}
        self.orientations_euler = {}
        self.correction = -np.eye(3)
        self.correction[0, 0] = 1
    

    def add_tag(self,id,x,y,z,theta_x,theta_y,theta_z):
        self.locations[id]=self.FeetToMetersTrsanslationVector(x,y,z)
        self.orientations[id]=self.eulerAnglesToRotationMatrix(theta_x,theta_y,theta_z)
        self.orientations_euler[id]=np.array([theta_x, theta_y, theta_z])

        
    # Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(self, theta_x, theta_y, theta_z):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta_x), -np.sin(theta_x)],
                        [0, np.sin(theta_x), np.cos(theta_x)]
                        ])

        R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                        [0, 1, 0],
                        [-np.sin(theta_y), 0, np.cos(theta_y)]
                        ])

        R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                        [np.sin(theta_z), np.cos(theta_z), 0],
                        [0, 0, 1]
                        ])

        R = np.matmul(R_z, np.matmul(R_y, R_x))

        return R.T
    

    def TranslationVector(self,x,y,z):
        return np.array([[x],[y],[z]])

    def FeetToMetersTrsanslationVector(self, x, y, z):
        return np.array([[x], [y], [z]])*0.3048

    def estimate_pose(self, tag_id, R, t):
        return (self.orientations[tag_id] @ (R.T @ -t)) + self.locations[tag_id]
    
    def estimate_euler_angles(self, tag_id, R, t):
        rot = (rotation_matrix_euler_angles(R) + self.orientations_euler[tag_id])/ np.pi * 180

        if rot[1] >= 0 and rot[1] <= 360:
            return rot - 360
        elif rot[1] < 0 and rot[1] >= -360:
            return rot
        else:
            print("ROTATION ERROR")
            return np.array([0.0, 0.0, 0.0])
        #return rotation_matrix_euler_angles(self.orientations[tag_id] @ R.T)

