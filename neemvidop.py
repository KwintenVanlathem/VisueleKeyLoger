import cv2
from tqdm import tqdm

vidtop = cv2.VideoCapture(2)
vidside = cv2.VideoCapture(0)

w = int(vidtop.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vidtop.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 8 #dubbele camera setup neemt in test aan 8.3 frames op per seconde, dus instellen op 8 zodat videos real-time lijken

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
outtop = cv2.VideoWriter('vid/toprot.mp4', fourcc, fps, (w,h))
outside = cv2.VideoWriter('vid/side.mp4', fourcc, fps, (w,h))

print("Frame W: " + str(w))
print("Frame H: " + str(h))
print("Frame FPS: " + str(fps))
	
s, frametop = vidtop.read()
s, frameside = vidside.read()

for i in tqdm(range(80)):
	s, frametop = vidtop.read()
	cv2.imshow("top", frametop)
	s, frameside = vidside.read()
	cv2.imshow("side", frameside)
	key = cv2.waitKey(1) & 0xFF

cv2.imshow("side", frameside)
print("START")
for i in tqdm(range(10000)):
	s, frametop = vidtop.read()
	outtop.write(frametop)
	s, frameside = vidside.read()
	outside.write(frameside)

	key = cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

vidtop.release()
vidside.release()
outtop.release()
outside.release()
cv2.destroyAllWindows()
