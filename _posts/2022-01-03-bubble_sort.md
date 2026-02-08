---
layout: post
tag-name: "algorithm"
title: Bubble Sort
date: 2022-01-03
---

## Bubble Sort
Bubble sort is sorting an array by repeatedly swapping the maximum element to the end of the array.

## Stability and Complexity
Bubble sort is  **stable**.

| Time complexity | Space complexity |
| --- | --- |
| $O(n^2)$ | $O(1)$ |


## Pseudo Code

```python
def bubble_sort(arr):
    for i in range(len(arr)):
        # sort arr[:len(arr)-i+1]
        for j in range(len(arr)-i-1):
            if arr[j+1]>arr[j]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
```

## Improved Bubble Sort
There are two ways to improve the effiency of  bubble sort:

### 1. early stop the swapping
If in an iteration, there is no swapping, we know that the array has already been sorted and there is no need to do the rest iterations.

```python
def bubble_sort_early_stop()
```