#!/usr/bin/env python
# coding: utf-8


from vedo import *
import numpy as np

def RotationMatrix(theta, axis_name):
    """ Calculate single rotation of $theta$ matrix around x,y or z
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

def createCoordinateFrameMesh():
    """Returns the mesh representing a coordinate frame
    Args:
      No input args
    Returns:
      F: vedo.mesh object (arrows for axis)
      
    """         
    _shaft_radius = 0.05
    _head_radius = 0.10
    _alpha = 1
    
    
    # x-axis as an arrow  
    x_axisArrow = Arrow(start_pt=(0, 0, 0),
                        end_pt=(1, 0, 0),
                        s=None,
                        shaft_radius=_shaft_radius,
                        head_radius=_head_radius,
                        head_length=None,
                        res=12,
                        c='red',
                        alpha=_alpha)

    # y-axis as an arrow  
    y_axisArrow = Arrow(start_pt=(0, 0, 0),
                        end_pt=(0, 1, 0),
                        s=None,
                        shaft_radius=_shaft_radius,
                        head_radius=_head_radius,
                        head_length=None,
                        res=12,
                        c='green',
                        alpha=_alpha)

    # z-axis as an arrow  
    z_axisArrow = Arrow(start_pt=(0, 0, 0),
                        end_pt=(0, 0, 1),
                        s=None,
                        shaft_radius=_shaft_radius,
                        head_radius=_head_radius,
                        head_length=None,
                        res=12,
                        c='blue',
                        alpha=_alpha)
    
    originDot = Sphere(pos=[0,0,0], 
                       c="black", 
                       r=0.10)


    # Combine the axes together to form a frame as a single mesh object 
    F = x_axisArrow + y_axisArrow + z_axisArrow + originDot
        
    return F
    
def forward_kinematics_for_animation(Phi, L1, L2, L3, L4):
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
    
    # Function implementation goes here
    
    radius_of_balls = 0.4
    
    # we assume the location of the first point is (3, 2, 0) wrt global frame
    
    trans_vec_01 = np.array([[3], [2], [0]])
    R_01 = RotationMatrix(Phi[0], axis_name = 'z')
    T_01 = getLocalFrameMatrix(R_01, trans_vec_01)
    
    trans_vec_12 = np.array([[0], [0], [L1]])
    R_12 = RotationMatrix(Phi[1], axis_name = 'y')
    T_12 = getLocalFrameMatrix(R_12, trans_vec_12)
    T_02 = T_01 @ T_12
    
    trans_vec_23 = np.array([[0], [0], [L2]])
    R_23 = RotationMatrix(Phi[2], axis_name = 'y')
    T_23 = getLocalFrameMatrix(R_23, trans_vec_23)
    T_03 = T_02 @ T_23
    
    trans_vec_34 = np.array([[0], [0], [L3]])
    R_34 = RotationMatrix(Phi[3], axis_name = 'y')
    T_34 = getLocalFrameMatrix(R_34, trans_vec_34)
    T_04 = T_03 @ T_34
    
    trans_vec_45 = np.array([[0], [0], [L4]])
    # Since the end effector has no orientation, the rotation
    # submatrix is just the identity matrix
    Identity_M = np.eye(3)
    T_45 = getLocalFrameMatrix(Identity_M, trans_vec_45)
    T_05 = T_04 @ T_45
    
    e = T_05[:, -1][:3]
    return T_01, T_02, T_03, T_04, T_05, e


def render_frames(Phi):
    # Lengths of  the parts   
    L1, L2, L3, L4 = [2, 5, 4, 3]
    #   Retrieve pose matrices and end-effector coordinates 
    #   for a given configuration   of  Phi =   [phi1, phi2, phi3, phi4]   
    # For the animation, phi1 is always 0
    
    T_01, T_02, T_03, T_04, T_05, e = forward_kinematics_for_animation(Phi, L1, L2, L3, L4)
    
    radius_of_balls = 0.4
    
    frames_in_order = []
    
    Frame1Arrows = createCoordinateFrameMesh()
    
    # Now, let's create a cylinder and add it to the local coordinate frame
    link1_mesh = Cylinder(r=radius_of_balls*2, 
                          height=L1, 
                          pos = (0,0,L1/2),
                          c="yellow", 
                          alpha=.5, 
                          axis=(0,0,1)
                          )
                          
    #sphere_1 = Sphere(r=radius_of_balls).pos(0,0,0).color("gray").alpha(.8)
    
    # Combine all parts into a single object 
    Frame1 = Frame1Arrows + link1_mesh

    # Transform the part to position it at its correct location and orientation 
    Frame1.apply_transform(T_01)
    
    frames_in_order.append(Frame1)
    
    Frame2Arrows = createCoordinateFrameMesh()
    
    link2_mesh = Cylinder(r=radius_of_balls, 
                          height=L2, 
                          pos = (0,0,L2/2),
                          c="red", 
                          alpha=.5, 
                          axis=(0,0,1)
                          )
                          
    sphere_2 = Sphere(r=radius_of_balls).pos(0,0,0).color("gray").alpha(.8)
    
    # Combine all parts into a single object 
    Frame2 = Frame2Arrows + link2_mesh + sphere_2

    # Transform the part to position it at its correct location and orientation 
    Frame2.apply_transform(T_02)
    
    frames_in_order.append(Frame2)
    
    Frame3Arrows = createCoordinateFrameMesh()
    
    link3_mesh = Cylinder(r=radius_of_balls, 
                          height=L3, 
                          pos = (0,0,L3/2),
                          c="cyan", 
                          alpha=.5, 
                          axis=(0,0,1)
                          )
                          
    sphere_3 = Sphere(r=radius_of_balls).pos(0,0,0).color("gray").alpha(.8)
    
    # Combine all parts into a single object 
    Frame3 = Frame3Arrows + link3_mesh + sphere_3

    # Transform the part to position it at its correct location and orientation 
    Frame3.apply_transform(T_03)
    
    frames_in_order.append(Frame3)
    
    Frame4Arrows = createCoordinateFrameMesh()
    
    link4_mesh = Cylinder(r=radius_of_balls, 
                          height=L4, 
                          pos = (0,0,L4/2),
                          c="pink", 
                          alpha=.5, 
                          axis=(0,0,1)
                          )
                          
    sphere_4 = Sphere(r=radius_of_balls).pos(0,0,0).color("gray").alpha(.8)
    
    # Combine all parts into a single object 
    Frame4 = Frame4Arrows + link4_mesh + sphere_4

    # Transform the part to position it at its correct location and orientation 
    Frame4.apply_transform(T_04)
    
    frames_in_order.append(Frame4)
                          
    sphere_5 = Sphere(r=radius_of_balls/4).pos(0,0,0).color("black").alpha(.8)
    
    # Combine all parts into a single object 
    Frame5 = sphere_5

    # Transform the part to position it at its correct location and orientation 
    Frame5.apply_transform(T_05)
    
    frames_in_order.append(Frame5)
    
    return frames_in_order

def main():
    # Phi = np.array([0, 45, 35, 25])
    # Set the limits of the graph x, y, and z ranges 
    axes = Axes(xrange=(0,25), yrange=(-2,10), zrange=(0,7))
    
    Phi = np.array([10, 30, 35, -45])
    scene = render_frames(Phi)
    
    plotter = Plotter()
    
    # show everything
    plotter.show(scene, axes, viewup="z").close()


if __name__ == '__main__':
    main()
