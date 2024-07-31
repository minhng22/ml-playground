# Amazon has recently established n distribution centers in a new location. They want to set up 2 warehouses to serve these distribution centers. Note that the centers and warehouses are all built along a straight line. A distribution center has its demands met by the warehouse that is closest to it. A logistics team wants to choose the location of the warehouses such that the sum of the distances of the distribution centers to their closest warehouses is minimized.
# Given an array dist_centers, that represent the positions of the distribution centers, return the minimum sum of distances to their closest warehouses if the warehouses are positioned optimally.
# Example
# Suppose dist_centers = [1, 2, 3].
# One optimal solution is to position the 2 warehouses at x1 =1 and x2
# = 2.
import random


def solution(arr):
    arr = sorted(arr)
    S, s, e, I = None, 0, 0, 0
    for i in range(len(arr)):
        med1 = arr[i//2]
        med2 = arr[i + ((len(arr)-i) // 2)]
        curr_S = sum([min(abs(arr[j] - med1), abs(arr[j] - med2)) for j in range(len(arr))])

        if not S or curr_S < S:
            S = curr_S
            s = i // 2
            e = i + ((len(arr)-i) // 2)
            I = i
    return S, s, e, I


def solution2(arr):
    arr = sorted(arr)
    s = len(arr) // 4
    e = len(arr) * 3 // 4
    S = 0
    for i in range(len(arr)):
        S += min(abs(arr[i] - arr[s]), abs(arr[i] - arr[e]))
    return S, s, e

def run_solution():
    for i in range(2, 10**3):
        A = [k for k in range(i+1)]
        S1, s1, e1, I1 = solution(A)
        S2, s2, e2 = solution2(A)
        if S1 < S2:
            print(f"not optimal at {i}. {s1}, {s2}, {e1}, {e2}")
    print("all good")


if __name__ == '__main__':
    run_solution()