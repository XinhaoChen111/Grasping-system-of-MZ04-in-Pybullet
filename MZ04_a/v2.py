import cv2

tgt = cv2.imread("./pybullet+opengl/prgb.png")
res = cv2.resize(tgt,(640, 480))

cv2.imwrite('./pybullet+opengl/bg640.png', res)