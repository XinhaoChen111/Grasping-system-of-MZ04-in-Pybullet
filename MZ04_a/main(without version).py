import numpy as np
import math
import random
import rospy
import time
import pybullet as p
import pybullet_data as pd

def get_EyesinHand(robotid):
    camera_Offset = [-0.0545,0,0.058]
    camera_UpVect = [-0.1,0,0.058]
    camera_Target = [-0.0545,0,0.458]
    J6_Position = p.getLinkState(robotid,6, computeForwardKinematics=1)[0]
    J6_OrientationQ = p.getLinkState(robotid, 6, computeForwardKinematics=1)[1]
    J6_RotM = p.getMatrixFromQuaternion(J6_OrientationQ)
    T6B = getTransfMat(J6_Position,J6_RotM)
    camera_PositionT = np.dot(T6B,trans2Mat(camera_Offset))
    camera_TgtPosT = np.dot(T6B,trans2Mat(camera_Target))
    camera_UpVectT = np.dot(T6B,trans2Mat(camera_UpVect))
    camera_UpVectC = mat2Trans(camera_UpVectT)
    
    camera_TgtPos = mat2Trans(camera_TgtPosT)
    camera_Position = mat2Trans(camera_PositionT)

    camera_UpVect = getVect(camera_UpVectC,camera_Position)
    view_matrix = p.computeViewMatrix(cameraEyePosition=camera_Position,
                                      cameraTargetPosition=camera_TgtPos,
                                      cameraUpVector=camera_UpVect)
    
    fov = 60
    aspect = float(640 / 480)
    near = 0.001
    far = 1000.0

    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    images = p.getCameraImage(width=1920,
                              height=1440,
                              viewMatrix=view_matrix,
                              projectionMatrix=proj_matrix,
                              renderer=p.ER_BULLET_HARDWARE_OPENGL)
    
    return images

def getTransfMat(offset, rotate):
    mat = np.array([
        [rotate[0], rotate[1], rotate[2], offset[0]], 
        [rotate[3], rotate[4], rotate[5], offset[1]], 
        [rotate[6], rotate[7], rotate[8], offset[2]],
        [0, 0, 0, 1.] 
    ])
    return mat

def getVect(camera_UpVectC,camera_Position):
    dVect = [camera_UpVectC[i] - camera_Position[i] for i in range(len(camera_UpVectC))]
    return dVect

def trans2Mat(offset):
    mat = np.array([
        [1, 0, 0, offset[0]], 
        [0, 1, 0, offset[1]], 
        [0, 0, 1, offset[2]],
        [0, 0, 0, 1.] 
    ])
    return mat

def mat2Trans(mat):
    offset = ([mat[0,3],mat[1,3],mat[2,3]])
    return offset




def get_TEnd(robotid):
    end_Offset = [0,0,0.1295] 
    J6_Position = p.getLinkState(robotid,6, computeForwardKinematics=1)[0]
    J6_OrientationQ = p.getLinkState(robotid, 6, computeForwardKinematics=1)[1]
    J6_RotM = p.getMatrixFromQuaternion(J6_OrientationQ)
    T6B = getTransfMat(J6_Position,J6_RotM)
    end_PositionT = np.matmul(T6B,trans2Mat(end_Offset))
    
    return end_PositionT

def addtargets(obj_number):
    targetId_all = []
    for i in range(obj_number):
        ret_x = random.uniform(-0.05, 0.05)
        ret_y = random.uniform(-0.05, 0.05)
        ret_z = random.uniform(-0.05, 0.05)
        ret_o = random.uniform(-1, 1)
        targetId = p.loadURDF("./model/table2/table2.urdf",
                           basePosition=[0.35+ret_x, ret_y, 0.55+ret_z], baseOrientation=[ret_o, 1, 0, 0], useFixedBase=False)
        #targetId=p.loadURDF("lego/lego.urdf",
        #                    basePosition=[0.35+ret_x, ret_y, 0.55+ret_z], baseOrientation=[ret_o, 1, 0, 0], useFixedBase=False, globalScaling=0.7)
        targetId_all.append(targetId)
    return targetId_all

def addtray2s():
    tray2Id_all = []
    for i in range(3):
        for j in range(2):
            tray2id = p.loadURDF("./model/hole/hole.urdf", basePosition=[0.25+j*0.1, -0.3-i*0.05, 0.32], globalScaling=1,
                             baseOrientation=startOrientation2, useFixedBase=True)
            tray2Id_all.append(tray2id)
    return tray2Id_all

def clutch(id,jointa,jointb,distance):
    p.setJointMotorControlArray(
        bodyUniqueId=id,
        jointIndices=[jointa,jointb],
        controlMode=p.POSITION_CONTROL,
        targetPositions=[distance,-2*distance],
        forces=[100,100]
    )

def moveit(robotid, targetid, dz, maxV):
    targetp, targeto = p.getBasePositionAndOrientation(targetid)
    targeto = p.getEulerFromQuaternion(targeto)
    jawp=[targetp[0],targetp[1],targetp[2]+dz]
    jawo=p.getQuaternionFromEuler([math.pi, 0, targeto[2]])
    jointPoses = p.calculateInverseKinematics(robotid, 6, jawp, jawo, maxNumIterations=20)
    for i in range(6):
        p.setJointMotorControl2(robotid, i+1, p.POSITION_CONTROL, jointPoses[i], force=5 * 240., maxVelocity=maxV)

def moveitforas(robotid, targetid, dz, maxV):
    targetp, targeto = p.getBasePositionAndOrientation(targetid)
    targeto = p.getEulerFromQuaternion(targeto)
    jawp=[targetp[0]+0.008,targetp[1],targetp[2]+dz]
    jawo=p.getQuaternionFromEuler([math.pi, 0, targeto[2]])
    jointPoses = p.calculateInverseKinematics(robotid, 6, jawp, jawo, maxNumIterations=20)
    for i in range(6):
        p.setJointMotorControl2(robotid, i+1, p.POSITION_CONTROL, jointPoses[i], force=5 * 240., maxVelocity=maxV)

if __name__ == "__main__":
    rospy.init_node('listener', anonymous=True)
    rate = rospy.Rate(1000)
    client = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pd.getDataPath())
    #p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    targetnum=8  #number of targets

    startOrientation1 = p.getQuaternionFromEuler([0, 0, 1.57])
    startOrientation2 = p.getQuaternionFromEuler([0, 0, 3.14])

    planeId = p.loadURDF("plane.urdf")
    # print(os.getcwd())
    mz04Id = p.loadURDF('./model/urdf/MZ04_with_jaw.urdf',basePosition=[0,0,0], useFixedBase=True)
    table1 = p.loadURDF("./model/table1/table1.urdf", basePosition=[0.6, -0.5, 0], baseOrientation=startOrientation1, useFixedBase=True)
    tray1id = p.loadURDF("tray/traybox.urdf", basePosition=[0.35, 0, 0.3], globalScaling=0.7, baseOrientation=startOrientation2, useFixedBase=True)
    #tray2id = p.loadURDF("./model/hole/hole.urdf", basePosition=[0.3, -0.3, 0.32], globalScaling=1, baseOrientation=startOrientation2, useFixedBase=True)
    tray2id=addtray2s()
    targetId_all=addtargets(targetnum)


    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


    mz04NumDofs = p.getNumJoints(mz04Id)
    assert mz04NumDofs == 10

    for i in range(9):
      p.changeDynamics(mz04Id, i, linearDamping=0, angularDamping=0)


    while True:
        for i in range(400): #400
                p.stepSimulation()
                
                moveit(mz04Id, tray1id, 0.4, 10)
                #for camera                
                time.sleep(0.001)
        for j in range(6):
            for i in range(200): #400
                p.stepSimulation()
                
                moveit(mz04Id, tray1id, 0.4, 10)
                #for camera                
                time.sleep(0.001)
            img = get_EyesinHand(mz04Id)
            time.sleep(0.01)
            for i in range(200):
                p.stepSimulation()
                clutch(mz04Id, 7, 8, 0.006)
                moveit(mz04Id, targetId_all[j], 0.25, 10)
                time.sleep(0.01)

            for i in range(200):
                p.stepSimulation()
                moveit(mz04Id, targetId_all[j], 0.18, 6)
                time.sleep(0.01)

            for i in range(200):
                p.stepSimulation()
                moveit(mz04Id, targetId_all[j], 0.138, 2)
                time.sleep(0.01)

            for i in range(100):
                p.stepSimulation()
                clutch(mz04Id, 7, 8, 0.015)
                time.sleep(0.01)

            for i in range(200):
                p.stepSimulation()
                moveit(mz04Id, tray1id, 0.3, 1.8)
                time.sleep(0.01)

            for i in range(200):
                p.stepSimulation()
                moveit(mz04Id, tray2id[j], 0.3, 1.8)
                time.sleep(0.01)

            for i in range(200):
                p.stepSimulation()
                moveit(mz04Id, tray2id[j], 0.25, 1.8)
                time.sleep(0.01)

            for i in range(200):
                p.stepSimulation()
                moveitforas(mz04Id, tray2id[j], 0.16, 1.8)
                time.sleep(0.01)

            for i in range(100):
                p.stepSimulation()
                clutch(mz04Id, 7, 8, 0)
                time.sleep(0.01)

            for i in range(200):
                p.stepSimulation()
                moveit(mz04Id, tray2id[j], 0.3, 10)
                time.sleep(0.01)

    p.disconnect()


