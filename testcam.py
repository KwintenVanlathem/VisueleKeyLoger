import cv2
from tqdm import tqdm

vid0 = cv2.VideoCapture(0)
vid1 = cv2.VideoCapture(2)
w = vid0.get(cv2.CAP_PROP_FRAME_WIDTH)
h = vid0.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Frame W: " + str(w))
print("Frame H: " + str(h))

for i in tqdm(range(1000)):
	succ, frame0 = vid0.read()
	succ, frame1 = vid1.read()
	cv2.imshow("camside", frame0)
	cv2.imshow("camtop", frame1)
	
	key = cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

vid0.release()
vid1.release()
cv2.destroyAllWindows()

