# SIFT로 특징점 및 디스크립터 추출(desc_sift.py)

import cv2
import numpy as np
import math

import matplotlib.pyplot as plt


img1 = cv2.imread(r"C:\Users\sungwoo\Downloads\attack_tineye\ramen2.jpg")
img2 = cv2.imread(r"C:\Users\sungwoo\Downloads\attack_tineye\target.jpg")

new_width = 800
new_height = 600

img1 = cv2.resize(img1, (new_width, new_height))
img2 = cv2.resize(img2, (new_width, new_height))


gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 서술자 추출기 생성 ---①
# detector = cv2.xfeatures2d.SIFT_create()


detector = cv2.ORB_create(nfeatures=153)

# 각 영상에 대해 키 포인트와 서술자 추출 ---②
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

# BFMatcher 생성, L1 거리, 상호 체크 ---③
matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# 매칭 계산 ---④
matches12 = matcher.match(desc1, desc2)

N = len(matches12)
print(N)
r = 3


print(len(kp1))
print(len(kp2))

# def rotate(x, y, theta):
#     x_k = x * math.cos(theta) + y * math.sin(theta)
#     y_k = - x * math.sin(theta) + y * math.cos(theta)

#     return x_k, y_k


# GX_1 = np.zeros(shape=(N, N, r))
# GY_1 = np.zeros(shape=(N, N, r))
# GX_2 = np.zeros(shape=(N, N, r))
# GY_2 = np.zeros(shape=(N, N, r))

# for i in range(N):
#     for j in range(N):
#         for k in range(r):
#             (x_i, y_i) = kp1[matches12[i].queryIdx].pt
#             (x_j, y_j) = kp1[matches12[j].queryIdx].pt

#             theta = k * math.pi / (4 * r)

#             x_i_k , y_i_k = rotate(x_i, y_i, theta)
#             x_j_k , y_j_k = rotate(x_j, y_j, theta)

#             if x_i_k >= x_j_k:
#                 GX_1[i][j][k] = 1
            
#             if y_i_k >= y_j_k:
#                 GY_1[i][j][k] = 1

            
#             (x_i, y_i) = kp2[matches12[i].trainIdx].pt
#             (x_j, y_j) = kp2[matches12[j].trainIdx].pt


#             x_i_k , y_i_k = rotate(x_i, y_i, theta)
#             x_j_k , y_j_k = rotate(x_j, y_j, theta)

#             if x_i_k >= x_j_k:
#                 GX_2[i][j][k] = 1
            
#             if y_i_k >= y_j_k:
#                 GY_2[i][j][k] = 1


# # print(GX_1, GY_1)
# # print(GX_2, GY_2)


# V_x = np.zeros(shape=(N, N, r))
# V_y = np.zeros(shape=(N, N, r))

# for i in range(N):
#     for j in range(N):
#         for k in range(r):
#             V_x[i][j][k] = (GX_1[i][j][k] != GX_2[i][j][k]) * 1
#             V_y[i][j][k] = (GY_1[i][j][k] != GY_2[i][j][k]) * 1


# S_x = np.zeros(shape=(N, r))
# S_y = np.zeros(shape=(N, r))

# for i in range(N):
#     sx = np.zeros(shape=r)
#     sy = np.zeros(shape=r)
#     for j in range(N):
#         sx += V_x[i][j]
#         sy += V_y[i][j]

#     S_x[i] = sx
#     S_y[i] = sy


# threshold = N / 2

# test = []
# for i in range(len(S_x)):
#     test.append(S_x[i][1])

# test.sort()


# def get_threshold(arr):
#     result = 0

#     a = arr[len(arr) - 1] / len(arr)
#     min_b = 0

#     for i in range(len(arr)):
#         x = (i + 1)
#         y = arr[i]
#         if min_b > y - a * x:
#             min_b = y - a * x
#             result = arr[i]

#     return result 

# plt.plot(test)
# plt.show()

# print(get_threshold(test))

# threshold = get_threshold(test)

# elements_greater_than_threshold = S_x > threshold

# output_array = np.any(elements_greater_than_threshold, axis=1)



# t = []
# f = []

# for i in range(len(output_array)):
#     if output_array[i] == False:
#         t.append(matches12[i])
#     else:
#         f.append(matches12[i])

# print(len(t), len(f))
    

# t_d = [i.distance for i in matches12]
# t_d.sort()

# plt.plot(t_d)
# plt.show()


sorted_matches = sorted(matches12, key=lambda x : x.distance)

res = cv2.drawMatches(img1, kp1, img2, kp2, sorted_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# 결과 출력 
cv2.imshow('true', res)

# res = cv2.drawMatches(img1, kp1, img2, kp2, matches12, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# # 결과 출력 
# cv2.imshow('false', res)

cv2.waitKey()
cv2.destroyAllWindows()