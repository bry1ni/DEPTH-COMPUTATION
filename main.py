import cv2
import numpy as np
import plotly.graph_objects as go

b = 5  # la distance entre les deux photos en cm
camera_to_objects = 30  # La distance entre la caméra et nos objets

"""NOS PARAMETRES INTRINSIC"""
mtx = np.load('camera_params/mtx.npy')
# mtx =  fx 0  Ox
#        0  fy Oy
#        0  0  1
fx = mtx[0][0]
fy = mtx[1][1]
Ox = mtx[0][2]
Oy = mtx[1][2]

"""NOS PHOTOS"""
img_l = cv2.imread('images/imageLEFT.jpg')  # query
imgl_gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

img_r = cv2.imread('images/imageRIGHT.jpg')  # train
imgr_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

"""KEYPOINTS COMPUTATION"""
sift = cv2.SIFT_create()
kpl, descl = sift.detectAndCompute(imgl_gray, None)
kpr, descr = sift.detectAndCompute(imgr_gray, None)

"""MATCHES COMPUTATION"""
match = cv2.BFMatcher()
matches = match.knnMatch(descl, descr, k=2)

good = [[m] for m, n in matches if m.distance < 0.5 * n.distance]


"""DRAW MATCHES"""
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(0, 0, 255),
                   flags=2)
img3 = cv2.drawMatchesKnn(img_l, kpl, img_r, kpr, good, None, **draw_params)

cv2.imshow('Result match', img3)
cv2.waitKey(0)
cv2.imwrite("results/Resultmatch_B4RANSAC.jpg", img3)

"""RANSAC"""
# work only with good matches using RANSAC
src_pts0 = np.float32([kpl[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts0 = np.float32([kpr[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask0 = cv2.findHomography(src_pts0, dst_pts0, cv2.RANSAC, 3, None, 100, 0.99)
matchesMask0 = mask0.ravel().tolist()

inlier_match0 = []
i = 0
for m in good:
    if matchesMask0[i]:
        inlier_match0.append(m)
    i = i + 1

"""DRAW MATCHES AFTER RANSAC"""
img3 = cv2.drawMatchesKnn(img_l, kpl, img_r, kpr, inlier_match0[:], None, **draw_params)
cv2.imshow('Result match 2', img3)
cv2.waitKey(0)
cv2.imwrite("results/Resultmatch_AFTERRANSAC.jpg", img3)

"""PROCESSING THE KEYPOINTS"""
# extarcting index of the good matches
# queryIdx refers to keypointsLEFT and trainIdx refers to keypointsRIGHT.
kpl_idx = [inlier_match0[idx][0].queryIdx for idx in range(len(inlier_match0))]
kpr_idx = [inlier_match0[idx][0].trainIdx for idx in range(len(inlier_match0))]

"""EXTRACTING x,y,z"""
# we need ul,vl ur,vr
ul = [kpl[kpl_idx[i]].pt[0] for i in range(len(kpl_idx))]
ur = [kpr[kpr_idx[i]].pt[0] for i in range(len(kpr_idx))]
vl = [kpl[kpl_idx[i]].pt[1] for i in range(len(kpl_idx))]
vr = [kpr[kpr_idx[i]].pt[1] for i in range(len(kpr_idx))]

# extract x,y,z
x = [(b * abs(ul[i] - Ox)) / abs(ul[i] - ur[i]) for i in range(len(ul))]  # x = b(ul - Ox) / (ul - ur)
y = [(b * fx * abs(vl[i] - Oy)) / (fy * abs(ul[i] - ur[i])) for i in range(len(ul))]  # y = b*fx(vl - Oy) / fy(ul - ur)
z = [(b * fx) / abs(ul[i] - ur[i]) for i in range(len(ul))]  # z = b*fx/ (ul - ur)

print(f"Profondeur moyenne des points: {np.mean(z)}")  # OUTPUT: 30.377985074146423 cm
print(f"Marge d'erreur: {np.mean(z) - camera_to_objects}")  # OUTPUT: 0.377985074146423


"""3D VISUALISATION"""
keypoints_3d = np.dstack([x, y, z])
# 3D Visualisation
fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color='red',
        colorscale='YlGn',
        opacity=0.8,
        reversescale=True
    )
)])
fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
fig.layout.scene.camera.projection.type = "orthographic"
fig.show()
