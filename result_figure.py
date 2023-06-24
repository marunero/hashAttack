import matplotlib.pyplot as plt

total = 15487

n = [2, 3, 5, 7]
  
hash_collision_ratio = [0, 3087 / total, 8853 / total, 9747 / total]
hash_decrement_avg = [0.0516, -27.58, -41.115, -43.032]

L2 = [3319, 2814, 4366, 3561]

plt.plot(n, L2, '-o')

plt.title("PDQ")
plt.xlabel("number of attack images")
plt.ylabel("L2 distance")

# 그래프 표시
plt.show()


## phash

# n = [2, 4, 6, 8]



# hash_collision_ratio = [4381 / total, 9644 / total, 10248 / total, 10145 / total]
# hash_decrement_avg = [-30.05, -42.2597, -43.76, -44.001]

# L2 = [2450, 3630, 3755, 3740]

# plt.plot(n, hash_decrement_avg, '-o')

# plt.title("phash")
# plt.xlabel("number of attack images")
# plt.ylabel("average hash decrement")

# # 그래프 표시
# plt.show()