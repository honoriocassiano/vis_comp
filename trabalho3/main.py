import imutils
import cv2
import numpy as np


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

	min_h = int(min(p1[1], p2[1]))
	max_h = int(max(p3[1], p4[1]))

	min_w = int(min(p1[0], p3[0]))
	max_w = int(max(p2[0], p4[0]))

	return (min_w, min_h, max_w, max_h)


def main():
	imgs = [ cv2.imread('data/%d.jpeg' % i) for i in range(1, 4) ]

	grays = [ cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs ]

	orb = cv2.ORB_create()

	kp_des = [ orb.detectAndCompute(i, None) for i in grays ]

	# Find matches
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = [ bf.match(kp_des[i][1], kp_des[i+1][1]) for i in range(len(kp_des) - 1) ]

	# Homographies
	Hs = [ None ] * len(matches)

	if len(matches) > 0:

		for i in range(len(matches)):

			src_pts = np.float32([ kp_des[i][0][m.queryIdx].pt for m in matches[i] ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp_des[i+1][0][m.trainIdx].pt for m in matches[i] ]).reshape(-1,1,2)

			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

			
			if i < (len(imgs) // 2):
				Hs[i] = M
			else:
				Hs[i] = np.linalg.inv(M)
			# Hs[i] = np.linalg.inv(M)

		# first = len(imgs) - 1
		# first = 0
		first = len(imgs) // 2
		curr_M = np.identity(3)

		curr_img = imgs[first]

		origin = np.array([0, 0, 1])

		for i in range(first-1, -1, -1):

			# Bounds of new image (transformed)
			(new_min_w, new_min_h, new_max_w, new_max_h) = get_size(imgs[i], curr_M @ Hs[i])

			# Translate matrix ()
			tr = np.identity(3)
			tr[:, 2] = [ -new_min_w, -new_min_h, 1 ]

			# Bounds of ranslated coordinates
			(new_min_w, new_min_h, new_max_w, new_max_h) = get_size(imgs[i], tr @ curr_M @ Hs[i])

			# Bounds of current panorama
			(curr_min_w, curr_min_h, curr_max_w, curr_max_h) = get_size(curr_img, tr)

			# New panorama bounds
			min_w = min(curr_min_w, new_min_w)
			max_w = max(curr_max_w, new_max_w)

			min_h = min(curr_min_h, new_min_h)
			max_h = max(curr_max_h, new_max_h)

			# Transform images
			new_image = cv2.warpPerspective(imgs[i], tr @ curr_M @ Hs[i], dsize=((max_w-min_w)+min_w, (max_h-min_h)+min_h))
			panorama = cv2.warpPerspective(curr_img, tr, dsize=((max_w-min_w)+min_w, (max_h-min_h)+min_h ))

			# Check overlap pixels
			mask1 = new_image.sum(axis=2).astype(bool)
			mask2 = panorama.sum(axis=2).astype(bool)

			mask = np.logical_and(mask1, mask2)

			# Set mean values for these pixels
			new_image[ mask ] //= 2
			panorama[ mask ] //= 2

			# Create new panorama
			new_panorama = new_image + panorama
			
			curr_img = new_panorama
			curr_M = tr @ curr_M @ Hs[i]

			origin = tr @ origin

		# curr_M = np.identity(3)
		tr_center = np.identity(3)

		# M_inv = np.linalg.inv(curr_M)

		# tr_center[:, 2] = curr_M[:, 2] / curr_M[2, 2]
		# tr_center[:, 2] = M_inv[:, 2] / M_inv[2, 2]
		# tr_center[:, 2] = np.linalg.inv(curr_M)[:, 2]
		# tr_center = np.linalg.inv(tr_center)

		# print(np.linalg.inv(M) @ np.array([0, 0, 1]))
		# tr_center[0, 2] = -origin[0]
		# tr_center[1, 2] = -origin[1]
		tr_center[:, 2] = origin

		print(tr_center)

		curr_M = np.identity(3)

		print(origin)

		
		for i in range(first, len(Hs)):

			# Bounds of new image (transformed)
			(new_min_w, new_min_h, new_max_w, new_max_h) = get_size(imgs[i+1], tr_center @ curr_M @ Hs[i])
			# (new_min_w, new_min_h, new_max_w, new_max_h) = get_size(imgs[i+1], curr_M @ Hs[i])



			print(new_min_w, new_min_h, new_max_w, new_max_h)

			# Translate matrix ()
			tr = np.identity(3)
			# tr[:, 2] = [ new_min_w, new_min_h, 1 ]
			tr[:, 2] = tr_center @ curr_M @ Hs[i] @ np.array([0, 0, 1])

			# print(tr)



			# tr = tr @ tr_center

			# Bounds of ranslated coordinates
			(new_min_w, new_min_h, new_max_w, new_max_h) = get_size(imgs[i+1], tr @ curr_M @ Hs[i])

			print(new_min_w)

			# Bounds of current panorama
			# (curr_min_w, curr_min_h, curr_max_w, curr_max_h) = get_size(curr_img, tr)
			(curr_min_w, curr_min_h, curr_max_w, curr_max_h) = get_size(curr_img, np.identity(3))

			# New panorama bounds
			min_w = min(curr_min_w, new_min_w)
			max_w = max(curr_max_w, new_max_w)

			min_h = min(curr_min_h, new_min_h)
			max_h = max(curr_max_h, new_max_h)

			# Transform images
			new_image = cv2.warpPerspective(imgs[i+1], tr @ curr_M @ Hs[i], dsize=((max_w-min_w)+min_w, (max_h-min_h)+min_h))
			panorama = cv2.warpPerspective(curr_img, np.identity(3), dsize=((max_w-min_w)+min_w, (max_h-min_h)+min_h ))

			# Check overlap pixels
			mask1 = new_image.sum(axis=2).astype(bool)
			mask2 = panorama.sum(axis=2).astype(bool)

			mask = np.logical_and(mask1, mask2)

			# Set mean values for these pixels
			new_image[ mask ] //= 2
			panorama[ mask ] //= 2

			# Create new panorama
			new_panorama = new_image + panorama
			
			curr_img = new_panorama
			curr_M = tr @ curr_M @ Hs[i]

		cv2.imwrite("panorama.jpg", curr_img)


if __name__ == '__main__':
	main()