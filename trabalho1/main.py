import cv2
import numpy as np


np.set_printoptions(suppress=True)


def main():

	points_img = [[ 496, 2185 ], [3940, 2236], [3125, 1060], [1055, 975]]

	# points_img = [[ 496, 3120 - 2185 ], [3940, 3120 - 2236], [3125, 3120 - 1060], [1055, 3120 - 975]]

	# points_world = [[0, 3120], [4160, 3120], [4160, 0], [0, 0]]

	# points_world = [[0+120, 240+120], [336+120, 240+120], [336+120, 0+120], [0+120, 0+120]]
	points_world = [[0, 240], [336, 240], [336, 0], [0, 0]]

	# 3120, 4160
	points_bounds = [[0, 3120], [4160, 3120], [4160, 0], [0, 0]]


	# vectors = np.array([
	# 	[
	# 		points_bounds[i][0] - points_img[i][0],
	# 		points_bounds[i][1] - points_img[i][1]
	# 	] for i in range(4) ]).T

	vectors_world = np.array([
		[
			points_world[i][0] - points_world[i-1][0],
			points_world[i][1] - points_world[i-1][1]
		] for i in range(4) ])

	vectors_square = np.array([
		[
			points_img[i][0] - points_img[i-1][0],
			points_img[i][1] - points_img[i-1][1]
		] for i in range(4) ])

	vectors_img = np.array([
		[
			points_bounds[i][0] - points_bounds[i-1][0],
			points_bounds[i][1] - points_bounds[i-1][1]
		] for i in range(4) ])

	denominator = np.linalg.norm(vectors_img, axis=0) ** 2

	projections = vectors_img * (np.diag(vectors_square.T @ vectors_img) / denominator)

	norms_img = np.linalg.norm(vectors_img, axis=1)
	norms_proj = np.linalg.norm(projections, axis=1)
	norms_world = np.linalg.norm(vectors_world, axis=1)
	norms_square = np.linalg.norm(vectors_square, axis=1)

	ratio_proj_img = norms_proj / norms_img
	ratio_world_img = norms_world / norms_square
	# print(vectors_img.shape)
	# print(vectors_square.shape)
	# print(norms)
	# print(projections)
	# print(norms_proj)
	# print(norms_img)
	print(ratio_world_img)
	print(ratio_proj_img)



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


	# points_img = [[0, 3120, 1], [4160, 3120, 1], [4160, 0, 1], [0, 0, 1]]).T
	# bounds = [ [*p, 1] for p in points_img ]

	# bounds_ = h_inv @ bounds
	# bounds_ = bounds_[:2] / bounds_[2]

	# max_b = bounds_.max(axis=1)
	# min_b = bounds_.min(axis=1)

	# print(bounds_)
	# print(max_b)
	# print(min_b)

	# min_bounds = [np.PINF, np.PINF]
	# max_bounds = [np.NINF, np.NINF]

	# # print(h)

	# for b in bounds:
	# 	temp = h_inv @ b
	# 	temp = temp[:2] / temp[2]
	# 	# temp = temp[:2] / temp[2]
	# 	# print(bounds)
	# 	# print(b, temp)
	# 	# print(src.shape)

	# 	print(temp)

	# 	if temp[0] < min_bounds[0]:
	# 		min_bounds[0] = temp[0]

	# 	if temp[0] > max_bounds[0]:
	# 		max_bounds[0] = temp[0]


	# 	if temp[1] < min_bounds[1]:
	# 		min_bounds[1] = temp[1]

	# 	if temp[1] > max_bounds[1]:
	# 		max_bounds[1] = temp[1]

	# scale = src.shape[0]/ (max_bounds[0] - min_bounds[0])
	transform = np.identity(3)

	# transform[:2, 2] = [1000, 500]

	dst = cv2.warpPerspective(src, transform, dsize=(src.shape[1], src.shape[0]))


	dst = cv2.warpPerspective(dst, h_inv, dsize=(src.shape[1], src.shape[0]))

	cv2.imwrite('abc.jpg', dst)


if __name__ == '__main__':
	main()