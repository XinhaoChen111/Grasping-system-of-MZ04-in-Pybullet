import numpy as np
import math
import random
import rospy
import time
import pybullet as p
import pybullet_data as pd
import cv2
import sys


from scipy.spatial.transform import Rotation as Rot

import random
from pyquaternion import Quaternion

sys.path.append('/home/ubuntu1804/Desktop/MZ04_p/pybullet+opengl/')
sys.path.append('/home/ubuntu1804/Desktop/MZ04_p/EfficientPose/')
sys.path.append('/home/ubuntu1804/Desktop/MZ04_p/dataset4icp/')
#from pybullet2opengl import pybullet2opengl
from pybullet2opengl import generateData
from evaluatePose import epEvaluate
from choose4icp import choose4icp
from depth2clouds import depth2clouds
from o3d_icp import o3d_icp



def addtargets(obj_number):
    targetId_all = []
    for i in range(obj_number):
        ret_x = random.uniform(-0.05, 0.05)
        ret_y = random.uniform(-0.05, 0.05)
        ret_z = random.uniform(-0.05, 0.05)
        ret_o = random.uniform(-1, 1)
        targetId = p.loadURDF("/home/ubuntu1804/Desktop/sources/pybullet+opengl/宝塔接头 3分14.SLDPRT/urdf/宝塔接头 3分14.SLDPRT.STL.urdf",
                             basePosition=[0.35+ret_x, ret_y, 0.45+ret_z], baseOrientation=[ret_o, 1, 0, 0], useFixedBase=False)
        # targetId = p.loadURDF("./model/table2/table2.urdf",
        #                    basePosition=[0.35+ret_x, ret_y, 0.45+ret_z], baseOrientation=[ret_o, 1, 0, 0], useFixedBase=False)
        #targetId=p.loadURDF(""/home/ubuntu1804/Desktop/sources/pybullet+opengl/宝塔接头 3分14.SLDPRT/urdf/宝塔接头 3分14.SLDPRT.STL.urdf"",
        #                    basePosition=[0.4+ret_x, ret_y, 0.5+ret_z], baseOrientation=[0, 1, 0, 0], useFixedBase=False, globalScaling=0.7)
        targetId_all.append(targetId)
    return targetId_all

def clutch(id,jointa,jointb,distance):
    p.setJointMotorControlArray(
        bodyUniqueId=id,
        jointIndices=[jointa,jointb],
        controlMode=p.POSITION_CONTROL,
        targetPositions=[distance,-2*distance],
        forces=[1000,1000] #force limit can debug #zuoyou yizhi
    )

#clutch luomu,dz=zhicha, 
def moveitfortarget(robotid, targetid, dz, maxV):
    targetp, targeto = p.getBasePositionAndOrientation(targetid)
    #tgtp=position tgto=ori,w.c.t[R,P,Y]
    targeto = p.getEulerFromQuaternion(targeto)
    #position compensation
    jawp=[targetp[0]-0.006*math.cos(targeto[2]),targetp[1]-0.006*math.sin(targeto[2]),targetp[2]+dz]
    jawo=p.getQuaternionFromEuler([math.pi, 0,targeto[2]])
    jointPoses = p.calculateInverseKinematics(robotid, 6, jawp, jawo)#maxNumIterations=20
    for i in range(6):
        p.setJointMotorControl2(robotid, i+1, p.POSITION_CONTROL, jointPoses[i], maxVelocity=maxV)

def moveit(robotid, targetid, dz, maxV):
    #origin
    targetp, targeto = p.getBasePositionAndOrientation(targetid)
    targeto = p.getEulerFromQuaternion(targeto)
    jawp=[targetp[0],targetp[1],targetp[2]+dz]
    jawo=p.getQuaternionFromEuler([math.pi, 0,targeto[2]])
    jointPoses = p.calculateInverseKinematics(robotid, 6, jawp, jawo)
    for i in range(6):
        p.setJointMotorControl2(robotid, i+1, p.POSITION_CONTROL, jointPoses[i], maxVelocity=maxV)

def moveitplus(robotid, targetid, dx,dy,dz, maxV):
    #dx dy dz w.c.t tgtid
    targetp, targeto = p.getBasePositionAndOrientation(targetid)
    targeto = p.getEulerFromQuaternion(targeto)
    jawp=[targetp[0]+dx,targetp[1]+dy,targetp[2]+dz]
    jawo=p.getQuaternionFromEuler([math.pi,1.57,targeto[2]])
    jointPoses = p.calculateInverseKinematics(robotid, 6, jawp, jawo)
    for i in range(5):
        p.setJointMotorControl2(robotid, i + 1, p.POSITION_CONTROL, jointPoses[i], maxVelocity=maxV)
    p.setJointMotorControl2(robotid, 6, p.POSITION_CONTROL, jointPoses[5]+1.57, maxVelocity=maxV)
    clutch(robotid, 7, 8, 0.016)
    #zitai +90

#try to improve success rate
def moveitforassemble(robotid, targetid, dx,dy,dz, maxV,toolo):
    targetp, targeto = p.getBasePositionAndOrientation(targetid)
    targeto = p.getEulerFromQuaternion(targeto)
    jawp = [targetp[0] + dx, targetp[1] + dy, targetp[2] + dz]
    jawo=p.getQuaternionFromEuler([0,1.57,0])
    jointPoses = p.calculateInverseKinematics(robotid, 6, jawp, jawo, maxNumIterations=20)
    for j in range(300):
        p.stepSimulation()
        #time sleep is important!
        time.sleep(0.01)
        for i in range(5):
            p.setJointMotorControl2(robotid, i + 1, p.POSITION_CONTROL, jointPoses[i], force=5 * 240., maxVelocity=maxV)
        p.setJointMotorControl2(robotid, 6, p.POSITION_CONTROL, jointPoses[5]+1.57, force=5 * 240., maxVelocity=2)
        time.sleep(0.01)
    jawp = [targetp[0] + dx, targetp[1] + dy-0.05, targetp[2] + dz]
    jawo = p.getQuaternionFromEuler([0, 1.57, 0])
    jointPoses = p.calculateInverseKinematics(robotid, 6, jawp, jawo, maxNumIterations=20)
    for j in range(300):
        p.stepSimulation()
        time.sleep(0.01)
        for i in range(5):
            p.setJointMotorControl2(robotid, i + 1, p.POSITION_CONTROL, jointPoses[i], force=5 * 240., maxVelocity=maxV)
        p.setJointMotorControl2(robotid, 6, p.POSITION_CONTROL, jointPoses[5] + 1.57, force=5 * 240., maxVelocity=2)
        time.sleep(0.01)
    jawp = [targetp[0] + dx, targetp[1] + dy-0.07, targetp[2] + dz]
    jawo = p.getQuaternionFromEuler([0, 1.57, 0])
    jointPoses = p.calculateInverseKinematics(robotid, 6, jawp, jawo, maxNumIterations=20)
    for j in range(300):
        p.stepSimulation()
        time.sleep(0.01)
        for i in range(5):
            p.setJointMotorControl2(robotid, i + 1, p.POSITION_CONTROL, jointPoses[i], force=5 * 240., maxVelocity=maxV)
        p.setJointMotorControl2(robotid, 6, p.POSITION_CONTROL, jointPoses[5] + 1.57, force=5 * 240., maxVelocity=2)
        time.sleep(0.01)

def get_pose(cubeOrn,cubePos):
    Rm = Rot.from_quat([cubeOrn[0], cubeOrn[1], cubeOrn[2], cubeOrn[3]])
    rotation_matrix = Rm.as_matrix()
    T_obj_world = np.identity(4) # generate 4*4 matrix
    T_obj_world[:3, :3] = rotation_matrix
    T_obj_world[0, 3] = cubePos[0]
    T_obj_world[1, 3] = cubePos[1]
    T_obj_world[2, 3] = cubePos[2]

    return T_obj_world

def get_truepose(cubeId,num):
    trueRT = []
    for i in range(num):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(cubeId[i])
        Rm = Rot.from_quat([cubeOrn[0], cubeOrn[1], cubeOrn[2], cubeOrn[3]])
        rotation_matrix = Rm.as_matrix()
        T_obj_world = np.identity(4) # generate 4*4 matrix
        T_obj_world[:3, :3] = rotation_matrix
        T_obj_world[0, 3] = cubePos[0]
        T_obj_world[1, 3] = cubePos[1]
        T_obj_world[2, 3] = cubePos[2]
        trueRT.append(T_obj_world)


    return trueRT

#with compensation, luomu waile zhihoude buchang 
def moveitforassemblec(robotid, targetid, dx,dy,dz, maxV,toolo):#not required yet
    targetp, targeto = p.getBasePositionAndOrientation(targetid)
    targeto = p.getEulerFromQuaternion(targeto)
    compeno=toolo-(-1.57)
    compend=0.3
    #jawp = [targetp[0] + dx, targetp[1] + dy, targetp[2] + dz]
    jawp=[targetp[0]+dx+compend*(1-math.cos(compeno)),targetp[1]+dy+compend*math.sin(compeno),targetp[2]+dz]
    jawo=p.getQuaternionFromEuler([0,1.57,0])
    jointPoses = p.calculateInverseKinematics(robotid, 6, jawp, jawo, maxNumIterations=20)
    for j in range(100):
        p.stepSimulation()
        time.sleep(0.01)
        print(compeno)
        for i in range(4):
            p.setJointMotorControl2(robotid, i + 1, p.POSITION_CONTROL, jointPoses[i], force=5 * 240., maxVelocity=maxV)
        p.setJointMotorControl2(robotid, 5, p.POSITION_CONTROL, jointPoses[4], force=5 * 240., maxVelocity=2)
        p.setJointMotorControl2(robotid, 6, p.POSITION_CONTROL, jointPoses[5]+1.57, force=5 * 240., maxVelocity=2)
        time.sleep(0.01)
    jawp = [targetp[0] + dx + compend * (1 - math.cos(compeno)), targetp[1] + dy-0.05 + compend * math.sin(compeno),
            targetp[2] + dz]
    jawo = p.getQuaternionFromEuler([0, 1.57, 0])
    jointPoses = p.calculateInverseKinematics(robotid, 6, jawp, jawo, maxNumIterations=20)
    for j in range(200):
        p.stepSimulation()
        time.sleep(0.01)
        print(compeno)
        for i in range(4):
            p.setJointMotorControl2(robotid, i + 1, p.POSITION_CONTROL, jointPoses[i], force=5 * 240., maxVelocity=maxV)
        p.setJointMotorControl2(robotid, 5, p.POSITION_CONTROL, jointPoses[4] + compeno, force=5 * 240., maxVelocity=2)
        p.setJointMotorControl2(robotid, 6, p.POSITION_CONTROL, jointPoses[5] + 1.57, force=5 * 240., maxVelocity=2)
        time.sleep(0.01)
    jawp = [targetp[0] + dx + compend * (1 - math.cos(compeno)), targetp[1] + dy-0.1 + compend * math.sin(compeno),
            targetp[2] + dz]
    jawo = p.getQuaternionFromEuler([0, 1.57, 0])
    jointPoses = p.calculateInverseKinematics(robotid, 6, jawp, jawo, maxNumIterations=20)
    for j in range(300):
        p.stepSimulation()
        time.sleep(0.01)
        print(compeno)
        for i in range(4):
            p.setJointMotorControl2(robotid, i + 1, p.POSITION_CONTROL, jointPoses[i], force=5 * 240., maxVelocity=maxV)
        p.setJointMotorControl2(robotid, 5, p.POSITION_CONTROL, jointPoses[4] + compeno, force=5 * 240., maxVelocity=2)
        p.setJointMotorControl2(robotid, 6, p.POSITION_CONTROL, jointPoses[5] + 1.57, force=5 * 240., maxVelocity=2)
        time.sleep(0.01)

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

def getCampos(robotid):
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
    Tcam = np.identity(4) # generate 4*4 matrix
    Tcam[:3, :3] = camera_PositionT[:3,:3]
    Tcam[:3, 3] = camera_PositionT[:3,3]
    Tcam =camera_PositionT
    return Tcam,view_matrix,proj_matrix

def get_EyesinHandC(robotid):
    camera_Offset = [-0.0545,0,0.058]
    camera_UpVect = [-0.1,0,0.05]
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
    print("test")
    # img4EP = images[2]
    # img4EPR = cv2.cvtColor(img4EP, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('./EfficientPose/testimage/data/0/rgb/0.png', img4EPR)

    return images


if __name__ == "__main__":
    rospy.init_node('listener', anonymous=True)
    rate = rospy.Rate(1000)
    client = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pd.getDataPath())
    #p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    targetnum=10  #number of targets

    startOrientation1 = p.getQuaternionFromEuler([0, 0, 1.57])
    startOrientation2 = p.getQuaternionFromEuler([0, 0, 3.14])
    startOrientation3 = p.getQuaternionFromEuler([0,0,0])

    planeId = p.loadURDF("plane.urdf")
    # print(os.getcwd())
    mz04Id = p.loadURDF('./model/urdf/MZ04_with_jaw.urdf',basePosition=[0,0,0], useFixedBase=True)
    table1 = p.loadURDF("./model/table1/table1.urdf", basePosition=[0.6, -0.7, 0], baseOrientation=startOrientation1, useFixedBase=True)
    tray1id = p.loadURDF("tray/traybox.urdf", basePosition=[0.35, 0, 0.3], globalScaling=0.7, baseOrientation=startOrientation2, useFixedBase=True)
    #tray2 is used to locate tray3,
    tray2id = p.loadURDF("./model/box/box.urdf", basePosition=[0.14, -0.3, 0.3], globalScaling=0.05,
                         baseOrientation=startOrientation2, useFixedBase=True)
    tray3id = p.loadURDF("./model/asmhole/asmhole.urdf", basePosition=[0.25, -0.28, 0.52], globalScaling=1, baseOrientation=startOrientation3, useFixedBase=True)
    targetId_all=addtargets(targetnum)



    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


    mz04NumDofs = p.getNumJoints(mz04Id)
    assert mz04NumDofs == 10

    for i in range(9):
      p.changeDynamics(mz04Id, i, linearDamping=0, angularDamping=0)
        ###  get T_camera_world
    theta = (-180 * 3.1415926) / 180.
    transformation = np.array(
        [[math.cos(theta), 0, math.sin(theta), -350], [0, 1, 0, 0], [-math.sin(theta), 0, math.cos(theta), 800], [0, 0, 0, 1]])
    # transformation = np.array(
    #     [[math.cos(theta), 0, math.sin(theta), 0], [0, 1, 0, 0], [-math.sin(theta), 0, math.cos(theta), 500], [0, 0, 0, 1]])

    theta1 = (-180 * 3.1415926) / 180.
    transformation2 = np.array(
        [[math.cos(theta1), 0, math.sin(theta1), 0], [0, 1, 0, 0], [-math.sin(theta1), 0, math.cos(theta1), 800], [0, 0, 0, 1]])

    theta2 = (180 * 3.1415926) / 180
    R_z = np.array([[math.cos(theta2), -math.sin(theta2), 0, 0],
                    [math.sin(theta2), math.cos(theta2), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                    ])
    T_camera_world = np.dot(R_z, transformation)   

    while True:
        #moveitfortarget(mz04Id, targetId_all[j], 0.25, 10), todo: for camera pos
        #the origin point of this func is J6, not gripper or camera
        img = get_EyesinHand(mz04Id)
        for j in range(targetnum):   #todo num of tgt or num of want to pick
            for i in range(300): #400
                p.stepSimulation()
                img = get_EyesinHand(mz04Id)
                moveit(mz04Id, tray1id, 0.5, 1000)
                #for camera                
                time.sleep(0.001)
            print("op")
            # imgt,projectionMatrix,viewMatrix = get_EyesinHand(mz04Id)
            # img_for_EP = imgt[2]
            # img_for_EP = cv2.cvtColor(img_for_EP, cv2.COLOR_BGR2RGB)
            # cv2.namedWindow('Camera Img', cv2.WINDOW_NORMAL)
            # cv2.imshow("Camera Img",img_for_EP)
            # cv2.waitKey(100)
            # print('ready4opengl')
            # getGRGB(targetnum,imgt,projectionMatrix,viewMatrix,targetId_all)
            Tcam,viewM,projM = getCampos(mz04Id)
            trueRT = get_truepose(targetId_all,targetnum)
            img = get_EyesinHand(mz04Id)
            generateData(targetId_all,targetnum,Tcam,viewM,projM)
            epEvaluate()
            choose4icp()
            depth2clouds(targetnum)
            o3d_icp(targetnum)
            # RT = np.load("./pybullet+opengl/dataset0/RT.npy")
            RT = np.load("./dataset4icp/RT_final.npy")
            trueRes = []
            for i in range(3):
                truetmp = np.dot(T_camera_world,RT[i])
                trueRes.append(truetmp)
                
            print("Tcam",T_camera_world)
            #opRT = np.dot(T_camera_world,RT)
            print("RT",trueRes)
            print("true",trueRT)

            #img = get_EyesinHandC(mz04Id)
            # for i in range(100):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     clutch(mz04Id, 7, 8, 0.006)
            #     moveitfortarget(mz04Id, targetId_all[j], 0.25, 10)
            #     time.sleep(0.001)

            # for i in range(150):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     moveitfortarget(mz04Id, targetId_all[j], 0.15, 5)
            #     time.sleep(0.001)

            # for i in range(150):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     moveitfortarget(mz04Id, targetId_all[j], 0.1378, 2) #0.133
            #     time.sleep(0.001)

            # for i in range(80):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     clutch(mz04Id, 7, 8, 0.016)
            #     time.sleep(0.001)

            # for i in range(100):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     moveit(mz04Id, tray1id, 0.3, 5)
            #     cv2.imshow("Camera Img",img_for_EP)
            #     cv2.waitKey(100)
            #     #for camera                
            #     time.sleep(0.001)


            #     #above is getting
            #     #below is asming
            # for i in range(500):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     moveitplus(mz04Id, tray2id,-0.035,0.09,0.35, 2.5)
            #     time.sleep(0.001)

            # for i in range(200):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     moveitplus(mz04Id, tray2id,-0.035,0.05,0.32, 1)
            #     time.sleep(0.001)

            # for i in range(200):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     moveitplus(mz04Id, tray2id,-0.035,0.03,0.312, 1)
            #     time.sleep(0.001)

            # for i in range(300):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     moveitplus(mz04Id, tray2id,-0.03,0,0.292, 1)
            #     time.sleep(0.001)
            # for i in range(500):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     moveitplus(mz04Id, tray2id,0,0.09,0.35, 2.5)
            #     time.sleep(0.01)

            # for i in range(300):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     moveitplus(mz04Id, tray2id,0,0.05,0.33, 1)
            #     time.sleep(0.01)

            # for i in range(300):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     moveitplus(mz04Id, tray2id,0,0.03,0.312, 1)
            #     time.sleep(0.01)

            # for i in range(500):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     moveitplus(mz04Id, tray2id,0.01,0,0.302, 1)
            #     time.sleep(0.01)

            _, targeto = p.getBasePositionAndOrientation(targetId_all[j])
            targeto = p.getEulerFromQuaternion(targeto)
            #genju luomu moduan jinxing buchang

            #moveit(mz04Id, tray2id, 0.25, 1.8)
            #moveitforassemble(mz04Id, tray2id, 0.315, 0.09, 0.2, 2, targeto[2])
            #moveitforassemblec(mz04Id, tray2id, -0.315, 0.09, 0.205, 2, targeto[2])

            # for i in range(1000):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     clutch(mz04Id, 7, 8, 0)
            #     time.sleep(0.001)
            #     #zhuazi fangsong

            # for i in range(500):
            #     p.stepSimulation()
            #     img = get_EyesinHand(mz04Id)
            #     moveitplus(mz04Id, tray2id,-0.035,0.09,0.35, 2.5)
            #     time.sleep(0.01)


    p.disconnect()


