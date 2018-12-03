import cv2
import numpy as np
import pygame as pg
from pygame.locals import *

run = True
# scale = 0
SCREEN = None
origin = None
img = None
# lines = [[]]
lines = [
	[ np.array([481, 669, 1]), np.array([170, 398, 1])],
	[ np.array([652, 510, 1]), np.array([338, 296, 1])],
	[ np.array([308, 515, 1]), np.array([613, 289, 1])],
	[ np.array([173, 398, 1]), np.array([474, 211, 1])]


	# [np.array([477, 395, 1]), np.array([651, 508, 1])],
	# [np.array([308, 518, 1]), np.array([482, 668, 1])],

	# [np.array([615, 290, 1]), np.array([725, 210, 1])],
	# [np.array([784, 383, 1]), np.array([884, 287, 1])]
]

ortho = [
	[ np.array([403, 76, 1]), np.array([289, 238, 1]) ],
	[ np.array([109, 122, 1]), np.array([466, 114, 1]) ],
	[ np.array([168, 282, 1]), np.array([410, 194, 1]) ],
	[ np.array([528, 152, 1]), np.array([405, 76, 1] )]


	# [np.array([249, 206, 1]), np.array([317, 141, 1])],
	# [np.array([317, 141, 1]), np.array([479, 140, 1])],

	# [np.array([ 98, 113, 1]), np.array([212,  76, 1])],
	# [np.array([212,  76, 1]), np.array([266, 108, 1])]
]


def calc_homography(img, lines):

	pass

# def pygame_to_cvimage(surface):
# 	"""Convert a pygame surface into a cv image"""
# 	cv_image = cv2.CreateImageHeader(surface.get_size(), cv2.IPL_DEPTH_8U, 3)
# 	image_string = surface_to_string(surface)
# 	cv2.SetData(cv_image, image_string)
# 	return cv_image


def cvimage_to_pygame(image):

	image_rgb = image[...,::-1]

	return pg.image.frombuffer(image_rgb.tostring(), image_rgb.shape[:2][::-1], "RGB")

def handle_events():

	global run, origin, lines
	
	for event in pg.event.get():
		t = event.type

		if t == QUIT:
			run = False

		if t == MOUSEBUTTONUP:
			# print(pg.mouse.get_pos())
			pos = np.array(pg.mouse.get_pos())

			print(pos)

			# # print(pos[(pos - img.shape[:2]) > 0])
			# # c1 = pos < origin
			# # c2 = pos > (origin + img.shape[:2][::-1])

			# # print(origin + img.shape[:2])

			# # print(pos)
			# # print(origin)
			# # print(c1)
			# # print(c2)

			# # if not pos[c1].shape[0] or not pos[c1].shape[0]:
			# if len(lines) < 5:

			# 	if len(lines[-1]) == 2:
			# 		lines.append([pos])
			# 	else:
			# 		lines[-1].append(pos)

			# 	print(lines)
			# 	# if len(lines) >= 2:
			# 	# 	# if len(lines) % 2 == 1:
			# 	# 	# 	pg.draw.lines(screen, (255)*3, False, lines[:-1])
			# 	# 	# else:
			# 	# 	pg.draw.lines(screen, (255, 255, 255), False, lines)

			# 	# 	print(lines)
			# 	# 	# print(lines)

		elif t == KEYDOWN:

			k = event.key

			if k == K_ESCAPE:
				run = False


def affine():
	global lines

	ls = []

	for l in lines:
		ls.append(np.cross(l[0], l[1]))

	x1 = np.cross(ls[0], ls[1])
	x2 = np.cross(ls[2], ls[3])

	l_inf = np.cross(x1, x2)

	# l1 = l1 / l1[2]
	# l2 = l2 / l2[2]
	# print(l_inf)

	hp = np.identity(3)
	hp[2] = l_inf / l_inf[2]

	# hp = np.linalg.inv(hp)

	# print(hp)

	# dst = cv2.warpPerspective(img, hp, dsize=(img.shape[1], img.shape[0]))
	dst = cv2.warpPerspective(img, hp, dsize=(img.shape[1], img.shape[0]))

	# transform = np.identity(3)
	# transform[0, 0] = transform[1, 1] = 1 / 10

	# dst = cv2.warpPerspective(dst, transform, dsize=(img.shape[1], img.shape[0]))

	# cv2.imwrite('teste.jpg', dst)
	# print(l1, l2, l_inf)
	# print(l_inf)
	return dst


def draw_lines(screen, lines):

	for l in lines:
		# TODO
		if len(l) == 2:
			pg.draw.line(screen, (255, 255, 255), l[0][:2], l[1][:2])


def metric(img):
	l1 = np.cross(ortho[0][0], ortho[0][1])
	m1 = np.cross(ortho[1][0], ortho[1][1])

	l2 = np.cross(ortho[2][0], ortho[2][1])
	m2 = np.cross(ortho[3][0], ortho[3][1])
	# l1 = ortho[0][0] * ortho[0][1]
	# m1 = ortho[1][0] * ortho[1][1]

	# l2 = ortho[2][0] * ortho[2][1]
	# m2 = ortho[3][0] * ortho[3][1]


	A = np.array([
					[ l1[0] * m1[0], l1[0] * m1[1] + l1[1] * m1[0] ],
					[ l2[0] * m2[0], l2[0] * m2[1] + l2[1] * m2[0] ]
				])

	B = np.array([-l1[1] * m1[1], -l2[1] * m2[1]])
	# B = np.array([l1[1] * m1[1], l2[1] * m2[1]])

	x = np.linalg.solve(A, B)

	# print(np.array([l1, m1]).T)
	# hht = np.zeros((3, 3))
	kkt = np.zeros((2, 2))
	kkt[0, 0] = x[0]
	kkt[0, 1] = kkt[1, 0] = x[1]
	kkt[1, 1] = 1

	# print(kkt)

	k = np.linalg.cholesky(kkt)
	

	# (d, u) = np.linalg.eig(kkt)

	# print(d)

	# d = np.sqrt(d)

	# k = u @ np.diag(d) @ u.T



	H = np.identity(3)

	H[:-1, :-1] = k

	# dst = cv2.warpPerspective(img, H, dsize=(img.shape[1], img.shape[0]))
	dst = cv2.warpPerspective(img, np.linalg.inv(H), dsize=(img.shape[1], img.shape[0]))
	
	cv2.imwrite('teste2.jpg', dst)


def main():

	global SCREEN, origin, img

	# img = cv2.imread('envelope_aleatorio.jpg')
	img = cv2.imread('piso.png')

	pg.init()  # Initialize pygame

	info = pg.display.Info()

	SCREEN = np.array([info.current_w // 2, info.current_w // 2])

	ratio_screen = (SCREEN[0] / SCREEN[1])
	ratio_img = (img.shape[1] / img.shape[0])

	if ratio_img >= ratio_screen:
		scale = SCREEN[0] / img.shape[1]
	else:
		scale = SCREEN[1] / img.shape[0]

	# print('abc', int(img.shape[1] * scale), int(img.shape[0] * scale))

	img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

	# draw_lines(None)

	# metric()




	# origin = (SCREEN - img.shape[:2][::-1]) / 2
	SCREEN = img.shape[:2][::-1]

	screen = pg.display.set_mode(img.shape[:2][::-1])
	
	# print(scale)

	tr = affine()

	surface = cvimage_to_pygame(tr)
	# surface = cvimage_to_pygame(tr)

	metric(tr)

	while run:
		screen.fill([0, 0, 0])

		handle_events()

		screen.blit(surface, (0, 0))

		draw_lines(screen, ortho)


		pg.display.update() # Update pygame display

		# break

if __name__ == '__main__':
	main()