import cv2
import numpy as np


# np.set_printoptions(suppress=True)

def get_size(img, M):
	(h, w) = img.shape[:-1]

	p1 = M @ np.array([0, 0, 1])
	p2 = M @ np.array([w, 0, 1])
	p3 = M @ np.array([0, h, 1])
	p4 = M @ np.array([w, h, 1])

	p1 = (p1 / p1[2])[:-1]
	p2 = (p2 / p2[2])[:-1]
	p3 = (p3 / p3[2])[:-1]
	p4 = (p4 / p4[2])[:-1]

	min_h = min(p1[1], p2[1])
	max_h = max(p3[1], p4[1])

	min_w = min(p1[0], p3[0])
	max_w = max(p2[0], p4[0])

	new_w = int(np.ceil(max_w - min_w))
	new_h = int(np.ceil(max_h - min_h))

	return (min_w, min_h, new_w, new_h)


def main():

	points_img = [[ 496, 2185 ], [3940, 2236], [3125, 1060], [1055, 975]]

	points_world = [[0, 240], [336, 240], [336, 0], [0, 0]]

	# 3120, 4160
	points_bounds = [[0, 3120], [4160, 3120], [4160, 0], [0, 0]]

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
	
	# print(ratio_world_img)
	# print(ratio_proj_img)

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

	h = np.append(x, 1).reshape((3, 3))
	h_inv = np.linalg.inv(h)

	src = cv2.imread('envelope_aleatorio.jpg')

	(min_w, min_h, new_w, new_h) = get_size(src, h_inv)

	transform = np.identity(3)
	transform[:2, 2] = [ -min_w, -min_h ]

	dst = cv2.warpPerspective(src, transform @ h_inv, dsize=(new_w, new_h))

	cv2.imwrite('envelope_aleatorio_ret.jpg', dst)


if __name__ == '__main__':
	main()