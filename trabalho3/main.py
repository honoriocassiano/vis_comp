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

	# new_w = max_w - min_w
	# new_h = max_h - min_h

	# return (min_w, min_h, new_w, new_h)
	return (min_w, min_h, max_w, max_h)


def main():
	# imgs = [ cv2.imread('data/%d.jpeg' % i) for i in range(1, 6) ]
	# imgs = [ cv2.imread('data/%d.jpeg' % i) for i in range(1, 3) ]
	imgs = [ cv2.imread('data/%d.jpeg' % i) for i in range(1, 6) ]
	# imgs = [ cv2.imread('panorama2.jpg'), cv2.imread('data/1.jpeg') ]

	grays = [ cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs ]

	orb = cv2.ORB_create()

	kp_des = [ orb.detectAndCompute(i, None) for i in grays ]

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	matches = [ bf.match(kp_des[i][1], kp_des[i+1][1]) for i in range(len(kp_des) - 1) ]

	# trs = [ None ] * len(matches)
	Ms = [ None ] * len(matches)
	bounds = [ None ] * len(matches)

	# print(type(matches[0]))
	# print(matches)
	# good = matches
	# for m,n in matches:
	# 	if m.distance < 0.7*n.distance:
	# 		good.append(m)

	# matches = sorted(matches, key=lambda x: x.distance)
	# cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
	# print(len(matches))
	if len(matches) > 0:

		for i in range(len(matches)):
			src_pts = np.float32([ kp_des[i][0][m.queryIdx].pt for m in matches[i] ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp_des[i+1][0][m.trainIdx].pt for m in matches[i] ]).reshape(-1,1,2)

			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

			(min_w, min_h, new_w, new_h) = bounds[i] = get_size(imgs[i], M)

			# tr = np.identity(3)
			# tr[:, 2] = [ -min_w, -min_h, 1 ]

			# trs[i] = tr
			Ms[i] = M
			# bounds[i] = imgs[i]



			# # tmp = cv2.warpPerspective(imgs[i], tr @ M, dsize=(new_w, new_h))
			# tmp = cv2.warpPerspective(imgs[i], tr @ M, dsize=(imgs[i+1].shape[0] - min_w, imgs[i+1].shape[1] - min_h))
			# tmp2 = cv2.warpPerspective(imgs[i+1], tr, dsize=(imgs[i+1].shape[0] - min_w, imgs[i+1].shape[1] - min_h))
			# # tmp = cv2.warpPerspective(imgs[i], M, dsize=(w, h))

			# mask1 = tmp.sum(axis=2).astype(bool)
			# mask2 = tmp2.sum(axis=2).astype(bool)

			# mask = np.logical_and(mask1, mask2)

			# # print(mask2)
			# # np.ma.array(tmp + tmp2, mask=)

			# # tmp3 = np.zeros(tmp2.shape)

			# t1 = tmp.copy()
			# t2 = tmp2.copy()

			# t1[ mask ] //= 2
			# t2[ mask ] //= 2

			# tmp3 = t1 + t2
			
			# # cv2.imwrite("result2_%d.jpg" % (i+1,), tmp)
			# # cv2.imwrite("result3_%d.jpg" % (i+1,), tmp2)
			# cv2.imwrite("panorama.jpg", tmp3)



		# first = len(imgs) // 2
		first = len(imgs) - 1
		curr_M = np.identity(3)
		curr_tr = np.identity(3)

		curr_img = imgs[first]
		
		

		# for i in range(first, len(Ms)):
		for i in range(first-1, -1, -1):

			# (min_w, min_h) = bounds[i][:2]

			# tr = np.identity(3)
			# tr[:, 2] = [ -min_w, -min_h, 1 ]
			# # curr_tr = curr_tr @ tr
			# curr_tr = tr

			# (curr_min_w, curr_min_h, curr_new_w, curr_new_h) = get_size(curr_img, curr_tr @ curr_M)
			# (min_w, min_h, new_w, new_h) = get_size(imgs[i], curr_M @ Ms[i])

			# (curr_min_w, curr_min_h, curr_max_w, curr_max_h) = get_size(curr_img, curr_tr @ curr_M)
			

			(new_min_w, new_min_h, new_max_w, new_max_h) = get_size(imgs[i], curr_M @ Ms[i])



			(min_w, min_h) = bounds[i][:2]

			tr = np.identity(3)
			tr[:, 2] = [ -new_min_w, -new_min_h, 1 ]
			# curr_tr = curr_tr @ tr
			curr_tr = tr



			(curr_min_w, curr_min_h, curr_max_w, curr_max_h) = get_size(curr_img, curr_tr @ curr_M)

			# print(min_w, min_h, new_w, new_h)
			min_w = min(curr_min_w, new_min_w)
			max_w = max(curr_max_w, new_max_w)

			min_h = min(curr_min_h, new_min_h)
			max_h = max(curr_max_h, new_max_h)


			tmp1 = cv2.warpPerspective(imgs[i], curr_tr @ curr_M @ Ms[i], dsize=( (max_w-min_w)+min_w, (max_h-min_h) ))
			# tmp1 = cv2.warpPerspective(imgs[i], curr_M @ Ms[i], dsize=( (max_w-min_w)+min_w, (max_h-min_h)+min_h ))
			tmp2 = cv2.warpPerspective(curr_img, curr_tr, dsize=((max_w-min_w)+min_w, (max_h-min_h) ))



			mask1 = tmp1.sum(axis=2).astype(bool)
			mask2 = tmp2.sum(axis=2).astype(bool)

			mask = np.logical_and(mask1, mask2)

			# print(mask2)
			# np.ma.array(tmp + tmp2, mask=)

			# tmp3 = np.zeros(tmp2.shape)

			tmp1[ mask ] //= 2
			tmp2[ mask ] //= 2

			tmp3 = tmp1 + tmp2
			
			# cv2.imwrite("result_abc_%d.jpg" % (i+1,), tmp2)
			# cv2.imwrite("result3_%d.jpg" % (i+1,), tmp2)
			cv2.imwrite("panorama.jpg", tmp3)

			curr_img = tmp3
			curr_M = tr @ curr_M @ Ms[i]


		# src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
		# dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

		# M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

		# print(M)

		# tmp = cv2.warpPerspective(img1, M, dsize=img1.shape[:-1])

		# cv2.imwrite("result.jpg", tmp)

if __name__ == '__main__':
	main()