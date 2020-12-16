import sys
import cv2
import numpy as np

def val1(v):
	global hu
	hu = v

def val2(v):
	global hl
	hl = v

def val3(v):
	global su
	su = v

def val4(v):
	global sl
	sl = v

def val5(v):
	global vu
	vu = v

def val6(v):
	global vl
	vl = v

path = str(sys.argv[1])
if str(sys.argv[2]) == "img":
	img = cv2.imread(path)
else:
	vid = cv2.VideoCapture(path)
	vid.set(cv2.CAP_PROP_POS_FRAMES, int(str(sys.argv[2])))
	s, img = vid.read()

hl, sl, vl = 0, 0, 0
hu, su, vu = 255, 255, 255

cv2.imshow("mask", img)
cv2.imshow("img", img)
cv2.createTrackbar("HL", "mask", hl, 180, val2)
cv2.createTrackbar("HH", "mask", hu, 180, val1)
cv2.createTrackbar("SL", "mask", sl, 255, val4)
cv2.createTrackbar("SH", "mask", su, 255, val3)
cv2.createTrackbar("VL", "mask", vl, 255, val6)
cv2.createTrackbar("VH", "mask", vu, 255, val5)
while(1):
	low = np.array([hl, sl, vl])
	high = np.array([hu, su, vu])
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, low, high)
	cv2.imshow("mask", mask)
	resulth = np.zeros(img.shape, dtype=np.uint8)
	resulth = cv2.bitwise_and(img, img, mask=mask)
	cv2.imshow("kleur", resulth)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()

print("HSV lower seg: ["+str(hl) +", "+str(sl)+", "+str(vl)+"]")
print("HSV upper seg: ["+str(hu) +", "+str(su)+", "+str(vu)+"]")

