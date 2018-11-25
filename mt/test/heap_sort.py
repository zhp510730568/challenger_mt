import math


def heap_sort(arr):
    pass


def parent(node_index):
    return math.floor((node_index - 1) / 2)


def child(node_index):
    return 2 * node_index + 1, 2 * node_index + 2


def swap(arr, i, j):
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp


def heapify(arr, i, length):
    tmp = arr[i]
    for index in range(2 * i + 1, length):
        max_index = index
        if index + 1 < length and arr[index] < arr[index + 1]:
            max_index += 1
        if arr[max_index] > tmp:
            arr[i] = arr[max_index]
            i = max_index
        else:
            break
    arr[i] = tmp


if __name__=='__main__':
    import time
    arr = [num for num in range(1000, 0, -1)]
    length = len(arr)
    start_time=time.time()
    for i in range(length - 1, -1, -1):
        heapify(arr, i, length)
    print(arr)
    for i in range(length - 1, -1, -1):
        swap(arr, i, 0)
        heapify(arr, 0, i)
    print(arr)
    print('delay time: ', time.time() - start_time)