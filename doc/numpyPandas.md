# NumPy and Pandas Note
Numpy 和 Pandas 使用手册，记录基础概念和常用操作。  
[NumPy 官方文档](https://www.numpy.org.cn/user/)  
[Pandas 官方文档](https://www.pypandas.cn/docs/)

# NumPy
NumPy包的核心是 ndarray 对象。它封装了python原生的同数据类型的 n 维数组，采用 c 语言实现后封装。

NumPy数组 和 原生Python Array（数组）之间有几个重要的区别：
+ NumPy 数组在创建时具有固定的大小，更改ndarray的大小将创建一个新数组并删除原来的数组;  
+ NumPy 数组中的元素都需要具有相同的数据类型，因此在内存中的大小相同。   
  (例外情况：Python的原生数组里包含了NumPy的对象的时候。)

## 基础知识
NumPy的主要对象是同构多维数组:  
+ ndarray.ndim  - 数组的轴（维度）的个数。
+ ndarray.shape - 数组的维度。对于有 n 行和 m 列的矩阵，shape 将是 (n,m)，ndim 将是2。
+ ndarray.size  - 数组元素的总数（等于 shape 的元素的乘积）。
+ ndarray.dtype - 一个描述数组中元素类型的对象。
+ ndarray.itemsize - 数组中每个元素的字节大小（等于 ndarray.dtype.itemsize）。
+ ndarray.data  - 该缓冲区包含数组的实际元素。

```python
    >>> import numpy as np
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> a.shape
    (3, 5)
    >>> a.ndim
    2
    >>> a.dtype.name
    'int64'
    >>> a.itemsize
    8
    >>> a.size
    15
    >>> type(a)
    <type 'numpy.ndarray'>
    >>> b = np.array([1, 2, 3])
    >>> b
    array([1, 2, 3])
    >>> b.shape
    (3,)
    >>> b.reshape(3, 1)
    array([[1],
           [2],
           [3]])
    >>> c = np.array([(1.2, 2.3), (3.4, 5.6)])
    >>> c
    array([[1.2, 2.3],
           [3.4, 5.6]])
    >>> c.shape
    (2, 2)
```

NumPy数组的几种特殊创建方式：zeros创建一个由0组成的数组；函数 ones创建一个完整的数组；  
函数empty 创建一个数组，其初始内容是随机的。  
此外，range函数按步长创建并返回数字类型的数组（而非列表）；linspace函数按元素数量创建数据数组。

```python
    >>> np.zeros((3,4))
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])
    >>> np.ones((2, 3, 4), dtype=np.int16)
    array([[[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]],    
           [[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]], dtype=int16)
    >>> np.arange(10, 30, 5)
    array([10, 15, 20, 25])
    >>> np.arange(0, 2, 0.3)
    array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
    >>> np.linspace(0, 2, 5)
    array([0. , 0.5, 1. , 1.5, 2. ])
```

NumPy数组采用 *矢量化* 技术，使数组操作更接近于标准的数学符号（通常，更容易正确编码数学结构）。






# Pandas






