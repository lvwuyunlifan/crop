import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
# x_data = np.float32(np.random.rand(2, 100)) # 随机输入
# y_data = np.dot([0.100, 0.200], x_data) + 0.300
#
# # 构造一个线性模型
# #
# b = tf.Variable(tf.zeros([1]))
# W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# y = tf.matmul(W, x_data) + b
#
# # 最小化方差
# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# # 初始化变量
# init = tf.initialize_all_variables()
#
# # 启动图 (graph)
# sess = tf.Session()
# sess.run(init)
#
# # 拟合平面
# for step in range(0, 201):
#     sess.run(train)
#     if step % 20 == 0:
#         print (step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]

# calculate the accuracy on validation set
def acc(y_true, y_pred):
    print('true: ',y_true)
    index = tf.reduce_any(y_true > 0.5, axis=-1)
    print('index2: ', index)
    res = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))

    index = tf.cast(index, tf.float32)
    print('index: ', index)
    res = tf.cast(res, tf.float32)
    print('res: ', res)
    return tf.reduce_sum(res * index) / (tf.reduce_sum(index) + 1e-7)

x = tf.constant([[True,  True], [False, False]])
print('a: ',tf.reduce_any(x))  # True
print('b: ',tf.reduce_any(x,0))  # [True, True]
print('c: ',tf.reduce_any(x,1))  # [True, False]
print('d: ',tf.reduce_any(x,-1))  # [True, False]



x_data = np.float32(np.random.rand(0, 10))
print(x_data)
y_true = np.array([[0.6,0.2,0.2]])
# y_true = [y_true, y_true, y_true]
y_pred = np.float32([1])
acc = acc(y_true, y_pred)
print("acc: ", acc)