import cv2
import numpy as np


np.set_printoptions(suppress=True)

def main():

	points_img = [[ 496, 2185 ], [3940, 2236], [3125, 1060], [1055, 975]]

	# points_img = [[ 496, 3120 - 2185 ], [3940, 3120 - 2236], [3125, 3120 - 1060], [1055, 3120 - 975]]

	# points_world = [[0, 3120], [4160, 3120], [4160, 0], [0, 0]]
	points_world = [[0, 240], [336, 240], [336, 0], [0, 0]]

	# 3120, 4160

	a = np.zeros((8, 8))
	b = np.zeros(8)

	for p in range(len(points_world)):

		i = p*2

		x = points_world[p][0]
		y = points_world[p][1]

		x_ = points_img[p][0]
		y_ = points_img[p][1]

		b[i] = x_
		b[i + 1] = y_

		a[i, :] = [ 
				x, y, 1,
				0, 0, 0,
				-x * x_, -y * x_ ]

		a[i + 1, :] = [ 
				0, 0, 0,
				x, y, 1,
				-x * y_, -y * y_ ]

	x = np.linalg.solve(a, b)

	# h = np.append(x, 1).reshape((3, 3))
	h = np.append(x, 1).reshape((3, 3))
	h_inv = np.linalg.inv(h)

	# print(x)

	# h[:2, :2] = h[:2, :2] * 2

	# print(h)
	# print(a)

	# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
	# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
	
	# M = cv.getPerspectiveTransform(pts1,pts2)
	# print(type(M))
	src = cv2.imread('envelope_aleatorio.jpg')


	bounds = [ 
		[0, 3120, 1], [4160, 3120, 1], [4160, 0, 1], [0, 0, 1]
	]
	# bounds = [ [*p, 1] for p in points_img ]

	min_bounds = [np.PINF, np.PINF]
	max_bounds = [np.NINF, np.NINF]

	# print(h)

	for b in bounds:
		temp = h_inv @ b
		temp = temp[:2] / temp[2]
		# temp = temp[:2] / temp[2]
		# print(bounds)
		# print(b, temp)
		# print(src.shape)

		print(temp)

		if temp[0] < min_bounds[0]:
			min_bounds[0] = temp[0]

		if temp[0] > max_bounds[0]:
			max_bounds[0] = temp[0]


		if temp[1] < min_bounds[1]:
			min_bounds[1] = temp[1]

		if temp[1] > max_bounds[1]:
			max_bounds[1] = temp[1]

	# print(min_bounds, max_bounds)
	print('')
	
	scale = src.shape[0]/ (max_bounds[0] - min_bounds[0])
	transform = np.identity(3)

	# transform[:2, 2] = [-min_bounds[0], -min_bounds[1]]

	# transform[:2, 2] = [-min_bounds[0] + 496, -min_bounds[1]]
	# transform[:2, 2] = [-min_bounds[0], -min_bounds[1]]
	# transform[0, 0] = transform[1, 1] = 10

	# print(transform)

	# transform[:, 2] = [1, 1, 1]

	h_inv = h_inv @ transform
	# print(src.shape)
	# abc = cv2.getPerspectiveTransform(np.array(points_img, np.float32), np.array(points_world, np.float32))

	# print(h, '\n')
	# print(abc)

	# src = cv2.imread('envelope_aleatorio.jpg')
	# dst = cv2.warpPerspective(src, h, dsize=src.shape[:2])
	# dst = cv2.warpPerspective(src, h, dsize=(src.shape[1],src.shape[0]) )

	dst = cv2.warpPerspective(src, h_inv, dsize=(src.shape[1],src.shape[0]) )
	# dst = cv2.warpPerspective(src, h, dsize=(
	# 	int(max_bounds[0] - min_bounds[0]),
	# 	int(max_bounds[1] - min_bounds[1])) )

	cv2.imwrite('abc.jpg', dst)


if __name__ == '__main__':
	main()