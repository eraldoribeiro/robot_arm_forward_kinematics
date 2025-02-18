#!/usr/bin/env python
# coding: utf-8


from vedo import *

def RotationMatrix(theta, axis_name):
    """ calculate single rotation of $theta$ matrix around x,y or z
        code from: https://programming-surgeon.com/en/euler-angle-python-en/
    input
        theta = rotation angle(degrees)
        axis_name = 'x', 'y' or 'z'
    output
        3x3 rotation matrix
    """

    c = np.cos(theta * np.pi / 180)
    s = np.sin(theta * np.pi / 180)
    
    if axis_name =='x':
        rotation_matrix = np.array([[1, 0,  0],
                                    [0, c, -s],
                                    [0, s,  c]])
    if axis_name =='y':
        rotation_matrix = np.array([[ c,  0, s],
                                    [ 0,  1, 0],
                                    [-s,  0, c]])
    elif axis_name =='z':
        rotation_matrix = np.array([[c, -s, 0],
                                    [s,  c, 0],
                                    [0,  0, 1]])
    return rotation_matrix


def getLocalFrameMatrix(R_ij, t_ij): 
    """Returns the matrix representing the local frame
    Args:
      R_ij: rotation of Frame j w.r.t. Frame i 
      t_ij: translation of Frame j w.r.t. Frame i 
    Returns:
      T_ij: Matrix of Frame j w.r.t. Frame i. 
      
    """             
    # Rigid-body transformation [ R t ]
    T_ij = np.block([[R_ij,                t_ij],
                     [np.zeros((1, 3)),       1]])
    
    return T_ij
    
def apply_transformation(X, H): 
    """transforms object using a compound transformation

    Args:
      X: 3 x N np.ndarray (float). It contains N points in 3-dimensional space 
                  in Cartesian coordinates. Each point is a column of the matrix.

      H: 4x4 Transformation matrix in homogeneous coordinates to be applied to the point set.     

    Returns:
      Y:  3 x N np.ndarray (float). It contains N points in 3-dimensional space 
                      in Cartesian coordinates. Each point is a column of the matrix.
      
    """    

    
    # Convert points to Homogeneous coords before transforming them
    
    # Apply transformation 
    Y = homogeneous2cartesian(H @ cartesian2homogeneous(X))
    
    # Convert points back to Cartesian coords before plotting
    
    return Y

def cartesian2homogeneous(X_c: np.ndarray) -> np.ndarray:
    """Converts the coordinates of a set of 3-D points from 
    Cartesian coordinates to homogeneous coordinates. 

    Args:
      X_c: M x N np.ndarray (float). It contains N points in M-dimensional space. 
           Each point is a column of the matrix.

    Returns:
      X_h: (M+1) x N np.ndarray (float) in homogeneous coords. It contains N points in (M+1)-dimensional space. 
           Each point is a column of the matrix.
      
    """    

    # Number of columns (number of points in the set). 
    ncols = X_c.shape[1]
    
    # Add an extra row of 1s in the matrix. 
    X_h = np.block([[X_c],
                   [ np.ones((1, ncols))]])

    return X_h
def homogeneous2cartesian(X_h: np.ndarray) -> np.ndarray:
    """Converts the coordinates of a set of 3-D points from 
    homogeneous coordinates to Cartesian coordinates. 

    Args:
      X_h: MxN np.ndarray (float) containing N points in homogeneous coords.  
           Each point is a column of the matrix.

    Returns:
      X_c: (M-1)xN np.ndarray (float) in Cartesian coords. 
           Each point is a column of the matrix.
      
    """    

    # Number of rows (dimension of points). 
    nrows = X_h.shape[0]
    
    # Divide each coordinate by the last to convert point set from homogeneous to Cartesian 
    # (using vectorized calculation for speed and concise code)
    X_c = X_h[0:nrows-1,:] / X_h[-1,:]

    return X_c
    
def forward_kinematics(Phi, L1, L2, L3, L4):
    """Calculate the local-to-global frame matrices,   
    and the location of the end-effector.
                    
    Args:
        Phi (4x1 nd.array):      Array containing the four joint angles
        L1, L2, L3, L4  (float): lengths of the parts of the robot arm.
                                 e.g., Phi = np.array([0, -10, 20, 0])
    
    Returns:
        T_01, T_02, T_03, T_04:  4x4 nd.arrays of local-to-global matrices
                                 for each frame.  
                                 e: 3x1 nd.array of 3-D coordinates, the location of the end-effector in space.          
        
    """
    
    radius_of_balls = 0.4
    
    # we assume the location of the first point is (3, 2, 0) wrt global frame
    
    trans_vec_01 = np.array([[3], [2], [0]])
    R_01 = RotationMatrix(Phi[0], axis_name = 'z')
    T_01 = getLocalFrameMatrix(R_01, trans_vec_01)
    
    trans_vec_12 = np.array([[L1 + radius_of_balls*2],[0], [0.0]])
    R_12 = RotationMatrix(Phi[1], axis_name = 'z')
    T_12 = getLocalFrameMatrix(R_12, trans_vec_12)
    T_02 = T_01 @ T_12
    
    trans_vec_23 = np.array([[L2 + radius_of_balls*2],[0], [0.0]])
    R_23 = RotationMatrix(Phi[2], axis_name = 'z')
    T_23 = getLocalFrameMatrix(R_23, trans_vec_23)
    T_03 = T_02 @ T_23
    
    trans_vec_34 = np.array([[L3 + radius_of_balls],[0], [0.0]])
    R_34 = RotationMatrix(Phi[3], axis_name = 'z')
    T_34 = getLocalFrameMatrix(R_34, trans_vec_34)
    T_04 = T_03 @ T_34
    
    e = T_04[:, -1][:3]
    return T_01, T_02, T_03, T_04, e

def main():
    # Lengths of  the parts   
    L1, L2, L3, L4 = [5, 8, 3, 0]
    #   Retrieve pose matrices and end-effector coordinates 
    #   for a given configuration of Phi = [phi1, phi2, phi3, phi4]
    Phi = np.array([-30, 50, 30, 0])
    T_01, T_02, T_03, T_04, e = forward_kinematics(Phi, L1, L2, L3, L4)
    print(T_04)
    print(e)

if __name__ == '__main__':
    main()
