l = 5
print("HELLO")
#     idxx = 2 * i
#     idxy = 2 * (i - 1)
#     e_odom = odoms[i] - odometry_estimation(x, i)
#     A[2 * i : 2 * i + 2, 2 * (i - 1) : 2 * (i - 1) + 4] = sqrt_inv_odom @ Hp
#     b[2 * i : 2 * i + 2] = sqrt_inv_odom @ e_odom

#     A[2 * (i + 1) : 2 * (i + 2), 2 * i : 2 * (i + 2)] = A_o
#     b[2 * (i + 1) : 2 * (i + 2)] = sqrt_inv_odom @ (odoms[i] - odometry_estimation(x, i))

for i in range(1, l + 1):
    print(f"{2 * i}:{2 * i + 2}, {2 * (i - 1)}:{2 * (i - 1) + 4}")

for i in range(l):
    print(f"{2 * (i + 1)}:{2 * (i + 2)}, {2 * i}:{2 * (i + 2)}")
