import numpy
import math


def get_max(array):
    return numpy.amax(numpy.array(array))


def get_average(array):
    return round(numpy.average(numpy.array(array)), 2)


def get_n_minus_one_subarrays(input, n, index, temp_indecies, temp_range_of_next_subarray, pool_type="max"):
    global start_index_of_the_last_subarray, last_subarray_dimension
    for j in range( n - 1):
        count = 0
        temp = []
        for a in range(temp_indecies, temp_range_of_next_subarray):
            temp.append(image[index][a])
            count += 1
        if pool_type is "avg":
            max_value = get_average(temp)
        else:
            max_value = get_max((temp))
        pooled_map.append(max_value)
        temp_indecies += interval
        temp_range_of_next_subarray += interval
    last_subarry = image[index][temp_indecies:]
    max_value = get_max(last_subarry)
    pooled_map.append(max_value)


def get_last_subarrays(start_index_of_the_last_subarray, temp_indecies, columns, pool_type = "max"):
    global last_subarray_dimension
    for j in range(n - 1):
        temp = []
        k = start_index_of_the_last_subarray
        for i in range(last_subarray_dimension):
            for a in range(temp_indecies, columns):
                temp.append(image[k][a])
            k += 1
        if pool_type is "avg":
            max_value = get_average(temp)
        else:
            max_value = get_max(temp)
        pooled_map.append(max_value)
        temp_indecies += interval
        columns += interval
        l = start_index_of_the_last_subarray
        last_subarray = []
        for start_index_of_the_last_subarraytem in range(last_subarray_dimension):
            last_subarray.append(image[l][temp_indecies:])
            l += 1
        last_max = get_max(last_subarray)
    pooled_map.append(last_max)


def get_final_result(one_d_list):
    index = 0
    pooled_list = []
    for i in range(m):
        temp_list = []
        for j in range(n):
            temp_list.append(one_d_list[index])
            index += 1
        pooled_list.append(temp_list)
    return numpy.array(pooled_list)


image = [
    [1, 2, 4, 2, 3],
    [4, 3, 0, 1, 5],
    [5, 0, 1, 4, 3]
]

"""
    : for m = 2
    : for n = 2
    : pool_type = "max"
"""

pooled_map = list()
m = 2
n = 2
image_height = 3
image_width = 5
range_of_next_subarray = math.ceil(image_width / n)
indecies = 0
start_index_of_the_last_subarray = 0
no_of_iterations_for_the_last_subarray = 0
interval = range_of_next_subarray
max_pooling = "max"
average_pooling = "avg"
for i in range(m - 1):
    get_n_minus_one_subarrays(image, n, i, indecies, range_of_next_subarray)
    no_of_iterations_for_the_last_subarray += 1
start_index_of_the_last_subarray = i + 1
last_subarray_dimension = image_height - no_of_iterations_for_the_last_subarray
get_last_subarrays(start_index_of_the_last_subarray, indecies, range_of_next_subarray)
print("falttened list", pooled_map)
ROI_POOLING = get_final_result(pooled_map)
print("The image was")
print(numpy.array(image))
print("The ROI max pooled map is({m}X{n})".format(m=m, n=n))
print(ROI_POOLING)

"""
    : for m = 2
    : for n = 3
    : pool_type = "avg"
"""

pooled_map = list()
m = 2
n = 3
image_height = 3
image_width = 5
range_of_next_subarray = math.ceil(image_width / n)
indecies = 0
start_index_of_the_last_subarray = 0
no_of_iterations_for_the_last_subarray = 0
interval = range_of_next_subarray
max_pooling = "max"
average_pooling = "avg"

for i in range(m - 1):
    get_n_minus_one_subarrays(image, n, i, indecies, range_of_next_subarray, average_pooling)
    no_of_iterations_for_the_last_subarray += 1
start_index_of_the_last_subarray = i + 1
last_subarray_dimension = image_height - no_of_iterations_for_the_last_subarray
get_last_subarrays(start_index_of_the_last_subarray, indecies, range_of_next_subarray, average_pooling)
print("falttened list", pooled_map)
ROI_POOLING = get_final_result(pooled_map)
print("The image was")
print(numpy.array(image))
print("The ROI average pooled map is({m}X{n})".format(m=m, n=n))
print(ROI_POOLING)









