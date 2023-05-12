import numpy as np

# 3차원 이미지 크기 지정
width = 256
height = 256
depth = 5  # 생성할 noise 개수

# # normal distribution N(0,1) 생성
# arr_list = [np.random.normal(loc=0, scale=1, size=(width, height))[:, :, np.newaxis] for i in range(depth)]
# arr = np.concatenate(arr_list, axis=2)

# # 확인을 위한 출력
# print(arr.shape)

a = np.random.normal(loc=0, scale=1, size=(width, height, 1))



np.array([np.random.normal(loc = 0, scale = 1, size = (32, 32, 1)) for i in range(depth)])

print(l.shape)