import imutils
import cv2
import numpy as np


def detectAndDescribe(image):
	# convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect keypoints in the image
	detector = cv2.FeatureDetector_create("SIFT")
	kps = detector.detect(gray)

	# extract features from the image
	extractor = cv2.DescriptorExtractor_create("SIFT")
	(kps, features) = extractor.compute(gray, kps)

	# convert the keypoints from KeyPoint objects to NumPy
	# arrays
	kps = np.float32([kp.pt for kp in kps])

	# return a tuple of keypoints and features
	return (kps, features)


def stitch(images):
	d = [ detectAndDescribe(i) for i in images ]
	M = [None] * (len(d) - 1)

	for i in range(len(M)):
		# match features between the two images
		M[i] = matchKeypoints(d[i][0], d[i+1][0],
			d[i][1], d[i+1][1])

	# # if the match is None, then there aren't enough matched
	# # keypoints to create a panorama
	# if M is None:
	# 	return None

def main():
	# imgs = [ cv2.imread('data/%d.jpeg' % i) for i in range(1, 6) ]

	img1 = cv2.imread('data/1.jpeg')
	img2 = cv2.imread('data/2.jpeg')

	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	orb = cv2.ORB_create()
	# kp = sift.detect(gray, None)
	# img1 = cv2.drawKeypoints(gray, kp, img1)
	# cv2.imwrite('sift_keypoints1.jpg', img1)

	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	# sift = cv2.ORB_create()
	# kp = sift.detect(gray, None)
	# img2 = cv2.drawKeypoints(gray, kp, img2)
	# cv2.imwrite('sift_keypoints2.jpg', img2)

	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)
	
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)

	# print(type(matches[0]))
	# print(matches)
	good = matches
	# for m,n in matches:
	# 	if m.distance < 0.7*n.distance:
	# 		good.append(m)

	# matches = sorted(matches, key=lambda x: x.distance)
	# cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
	# print(len(matches))
	if len(matches) > 0:
		# print(good[0])

		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

		print(M)

		tmp = cv2.warpPerspective(img1, M, dsize=img1.shape[:-1])

		cv2.imwrite("result.jpg", tmp)

if __name__ == '__main__':
	main()