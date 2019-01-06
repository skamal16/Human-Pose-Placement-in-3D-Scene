import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import time

## A set of classes to assist in generating objects to for an example environment for testing:
    
# 3D box object: Box(grid, x_coord = 0, y_coord = 0, z_coord = 0, width = 1, depth = 1, height = 1)
class Box():

    def __init__(self, grid, x_coord = 0, y_coord = 0, z_coord = 0, width = 1, depth = 1, height = 1):

        x, y, z = grid
        self.voxels = (x <= x_coord + width) & (x > x_coord) & (y <= y_coord + depth) & (y > y_coord) & (z <= z_coord + height) & (z > z_coord)

    def get_voxels(self):

        return self.voxels

# 3D chair object: Chair(grid, x = 0, y = 0, z = 0, size = 1, color = 'white')
class Sofa():

    def __init__(self, grid, x = 0, y = 0, z = 0, axis = 0):
        self.x = x
        self.y = y
        self.z = z
        self.grid = grid
        self.axis = axis

    def get_voxels(self):

        grid = self.grid
        axis = self.axis

        x, y, z = self.x, self.y, self.z

        if axis == 0:
            box1 = Box(grid, x, y, z, 6, 6, 2)
            box2 = Box(grid, x, y, z + 2, 2, 6, 2)
        elif axis == 1:
            box1 = Box(grid, x, y, z, 6, 6, 2)
            box2 = Box(grid, x, y + 4, z + 2, 6, 2, 2)
        elif axis == 2:
            box1 = Box(grid, x, y, z, 6, 6, 2)
            box2 = Box(grid, x, y, z + 2, 6, 2, 2)
        elif axis == 3:
            box1 = Box(grid, x, y, z, 6, 6, 2)
            box2 = Box(grid, x + 4, y, z + 2, 2, 6, 2)

        voxels = box1.get_voxels() + box2.get_voxels()

        return voxels
    
## A set of example pose classes, axis (0 - 3) defines direction pose is facing:

# Sitting Pose: SittingPose(grid, x, y, z, axis = 0)
class SittingPose():

    def __init__(self, grid, x, y, z, axis = 0):

        self.x = x
        self.y = y
        self.z = z
        self.grid = grid
        self.axis = axis

    def get_voxels(self):
        
        if(self.axis == 0):
            base, support = self.sitting_pose_0(self.grid, self.x, self.y, self.z)
        elif(self.axis == 1):
            base, support = self.sitting_pose_1(self.grid, self.x, self.y, self.z)
        elif(self.axis == 2):
            base, support = self.sitting_pose_2(self.grid, self.x, self.y, self.z)
        elif(self.axis == 3):
            base, support = self.sitting_pose_3(self.grid, self.x, self.y, self.z)
        return base, support

    def sitting_pose_0(self, grid, x, y, z):
        base1 = Box(grid, x, y, z, 4, 2, 4).get_voxels()
        base2 = Box(grid, x, y + 2, z + 2, 4, 2, 2).get_voxels()
        base3 = Box(grid, x, y + 4, z + 2, 4, 2, 4).get_voxels()

        base = base1 + base2 + base3

        support1 = Box(grid, x, y + 4, z + 1, 4, 2, 1).get_voxels()
        support2 = Box(grid, x, y + 6, z + 2, 4, 1, 2).get_voxels()

        support = support1 + support2

        return base, support

    def sitting_pose_1(self, grid, x, y, z):
        base1 = Box(grid, x, y, z, 2, 4, 4).get_voxels()
        base2 = Box(grid, x + 2, y, z + 2, 2, 4, 2).get_voxels()
        base3 = Box(grid, x + 4, y, z + 2, 2, 4, 4).get_voxels()

        base = base1 + base2 + base3

        support1 = Box(grid, x + 4, y, z + 1, 2, 4, 1).get_voxels()
        support2 = Box(grid, x + 6, y, z + 2, 1, 4, 2).get_voxels()

        support = support1 + support2

        return base, support

    def sitting_pose_2(self, grid, x, y, z):
        base1 = Box(grid, x, y, z, 2, 4, 4).get_voxels()
        base2 = Box(grid, x - 2, y, z + 2, 2, 4, 2).get_voxels()
        base3 = Box(grid, x - 4, y, z + 2, 2, 4, 4).get_voxels()

        base = base1 + base2 + base3

        support1 = Box(grid, x - 4, y, z + 1, 2, 4, 1).get_voxels()
        support2 = Box(grid, x - 5, y, z + 2, 1, 4, 2).get_voxels()

        support = support1 + support2

        return base, support

    def sitting_pose_3(self, grid, x, y, z):
        base1 = Box(grid, x, y, z, 4, 2, 4).get_voxels()
        base2 = Box(grid, x, y - 2, z + 2, 4, 2, 2).get_voxels()
        base3 = Box(grid, x, y - 4, z + 2, 4, 2, 4).get_voxels()

        base = base1 + base2 + base3

        support1 = Box(grid, x, y - 4, z + 1, 4, 2, 1).get_voxels()
        support2 = Box(grid, x, y - 5, z + 2, 4, 1, 2).get_voxels()

        support = support1 + support2

        return base, support

# Algorithm that returns a 3D ndrray of 0 and 1 where 1 represents whether a Pose object can be instantiated without breaking free space and support constraints:
def check_availability(grid, room, axis):

    shape = room.shape

    free_space = np.zeros(shape)
    support_space = np.zeros(shape)

    _, support = SittingPose(grid, shape[0]/2, shape[1]/2, shape[2]/2, axis).get_voxels()
    no_support_blocks = np.sum(support)

    room_flat = np.ndarray.flatten(room * 1)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                voxels, support = SittingPose(grid, i, j, k, axis).get_voxels()

                voxel_flat = np.ndarray.flatten(voxels * 1)
                #print(voxel_flat.shape)
                #print(room_flat.shape)
                free_space[i][j][k] =  (np.correlate(voxel_flat, room_flat) == 0)

                #print("support blocks:", no_support_blocks)
                support_space[i][j][k] = (np.correlate(np.ndarray.flatten(support * 1), room_flat) == no_support_blocks)

                #print("correlation:", np.correlate(np.ndarray.flatten(support * 1), np.ndarray.flatten(room * 1)))

                #print(i + 1, j + 1, k + 1)
    print("sum of free space:", np.sum(free_space))
    print("sum of support space:", np.sum(support_space))
    
    availability = support_space * free_space

    return availability

def corrbool(x, y):
    # correlation of 2 boolean arrays

    cor = np.sum(np.multiply(x, y), -1)

    return cor

def check_availability_vectorized(grid, room, axis):

    shape = room.shape

    free_space = np.zeros(shape)
    support_space = np.zeros(shape)

    _, support = SittingPose(grid, shape[0]/2, shape[1]/2, shape[2]/2, axis).get_voxels()
    no_support_blocks = np.sum(support)

    room_flat = np.ndarray.flatten(room * 1)

    support_mat = np.zeros((shape[0], shape[1], shape[2], room_flat.shape[0]))
    voxel_mat = np.zeros(support_mat.shape)

    start = time.time()
    print("loop")

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                voxels, support = SittingPose(grid, i, j, k, axis).get_voxels()
                voxel_mat[i][j][k] = np.ndarray.flatten(voxels * 1)
                support_mat[i][j][k] = np.ndarray.flatten(support * 1)


    end = time.time()
    print(end - start)
    start = time.time()
    print("vectorized algo")

    free_space = (corrbool(voxel_mat, room_flat) == 0)
    support_space = (corrbool(support_mat, room_flat) == no_support_blocks)
    
    availability = support_space * free_space

    end = time.time()
    print(end - start)

    print("sum of free space:", np.sum(free_space))
    print("sum of support space:", np.sum(support_space))

    return availability

def check_availability_GPU(grid, room):

    shape = room.shape

    print(shape)
    
    availability = tf.Variable(np.zeros(shape), name = 'availability')

    function = tf.global_variables_initializer()

    for i in range(shape[0]):
        for j in range(shape[0]):
            for k in range(shape[0]):
                 pose, support = SittingPose(grid, i, j, k, axis).get_voxels()
                 availability[i][j][k] = tf.contrib.metrics.streaming_pearson_correlation(tf.reshape(pose * 1, -1), tf.reshape(room * 1, -1)) != 0
                 print(i, j, k)

    with tf.Session() as session:
        return session.run(availability)

def get_pose_spawn(grid, room):
    
    #start = time.time()
    #print("vectorized")
    #availability = check_availability_vectorized(grid, room, axis)
    #end = time.time()
    #print(end - start)

    start = time.time()

    Pose_spawn = np.zeros(room.shape)

    for axis in range(4):
        availability = check_availability(grid, room, axis)

        pose_spawn = np.zeros(room.shape)

        for i in range(room.shape[0]):
            for j in range(room.shape[1]):
                for k in range(room.shape[2]):
                    if(availability[i][j][k] != 0):
                        pose, support = SittingPose(grid, i, j, k, axis).get_voxels()
                        pose_spawn = pose_spawn + pose

        Pose_spawn += pose_spawn

    end = time.time()
    print("Time taken for pose placement algo: ")
    print(end - start)

    return Pose_spawn > 0

def main():
    # prepare some coordinates
    grid = np.indices((25, 25, 25))

    # draw objects
    #chair = Chair(grid, 10, 10, 5, size = 1, color = 'blue')

    floor = Box(grid, 0, 0, 0, 50, 50, 1)

    wall1 = Box(grid, 0, 0, 0, 50, 1, 50)
    wall2 = Box(grid, 0, 0, 0, 1, 50, 50)
    wall3 = Box(grid, 0, 48, 0, 50, 1, 50)
    wall4 = Box(grid, 48, 0, 0, 1, 50, 50)

    pose1 = SittingPose(grid, 15, 15, 15)

    #chair_voxels, chair_color = chair.get_data()

    sofas = (
        Sofa(grid, 1, 1, 1, 0).get_voxels() + Sofa(grid, 9, 9, 1, 1).get_voxels() + Sofa(grid, 17, 1, 1, 3).get_voxels() + 
        Sofa(grid, 1, 30, 1, 0).get_voxels() + Sofa(grid, 9, 38, 1, 1).get_voxels() + Sofa(grid, 17, 30, 1, 3).get_voxels() + Sofa(grid, 9, 22, 1, 2).get_voxels() +
        Sofa(grid, 21, 15, 1, 0).get_voxels() + Sofa(grid, 29, 23, 1, 1).get_voxels() + Sofa(grid, 37, 15, 1, 3).get_voxels() + Sofa(grid, 29, 7, 1, 2).get_voxels()
        )

    floor = floor.get_voxels()

    walls = wall1.get_voxels() + wall1.get_voxels() + wall2.get_voxels() + wall3.get_voxels() + wall4.get_voxels()

    pose1_voxels, pose1_support_voxels = pose1.get_voxels()

    # combine the objects into a single boolean array

    room_display = sofas + floor

    room = room_display + walls

    pose_voxels = pose1_voxels
    pose_support_voxels = pose1_support_voxels

    poses = pose_voxels + pose_support_voxels

    pose_spawn = get_pose_spawn(grid, room)

    final = room_display + pose_spawn

    #print(room.shape, '\n break \n', poses.shape)

    #result = np.correlate(np.ndarray.flatten(poses * 1), np.ndarray.flatten(room * 1))

    #print(result)

    # set the colors of each object
    #room_colors = np.empty(room.shape, dtype=object)

    pose_colors = np.empty(poses.shape, dtype=object)

    final_colors = np.empty(final.shape, dtype=object)

    pose_colors[pose_voxels] = 'red'
    pose_colors[pose_support_voxels] = 'green'

    final_colors[room] = 'blue'
    final_colors[pose_spawn] = 'yellow'

    # and plot everything
    fig = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()

    ax = fig.gca(projection='3d')
    ax2 = fig2.gca(projection='3d')
    ax3 = fig3.gca(projection='3d')
    ax4 = fig4.gca(projection='3d')

    ax.voxels(room_display, facecolors='blue', edgecolor='k')
    ax2.voxels(poses, facecolors=pose_colors, edgecolor='k')
    ax3.voxels(pose_spawn, facecolors='yellow', edgecolor='k')
    ax4.voxels(final, facecolors=final_colors, edgecolor='k')

    plt.show()

main()