#Kwinten Vanlathem; visuele keylogger
import cv2
import sys
import time
from tqdm import tqdm

import functies

#PREPROCESSING:


#werk vanuit dezelfde folder
folder = str(sys.argv[1])

#draai top om
functies.draai180(folder+"/toprot.mp4", folder+"/top.mp4")

#save side background
sideback = functies.neemBackground(folder+"/side.mp4", 20, folder+"/sideback.png")
#save top background
functies.neemBackground(folder+"/top.mp4", 20, folder+"/topback.png")

#define bounding boxes voor elke toets
functies.maakToetsBoxFile(folder+"/keybox.txt", folder+"/topback.png") #comment out indien vorige hergebruiken
lijstKeyBox = functies.leesKeyboundIn(folder+"/keybox.txt")

#side masker toetsenbord
functies.maakToetsGrensFile(folder+"/key.txt", folder+"/sideback.png") #comment out indien vorige hergebruiken
poly = functies.leesToetsgrensIn(folder+"/key.txt")
sidemasker = functies.contactSide(sideback, poly)

#PER FRAME PROCESSING:
vidtop = cv2.VideoCapture(folder+"/top.mp4")
vidside = cv2.VideoCapture(folder+"/side.mp4")

offset = 4	#pas aan indien de twee video's niet synchroon lopen
vidtop.set(cv2.CAP_PROP_POS_FRAMES, offset)
pbar = tqdm(total = int(vidside.get(cv2.CAP_PROP_FRAME_COUNT))-offset)
for i in range(int(vidside.get(cv2.CAP_PROP_FRAME_COUNT))-offset):
	s, ftop = vidtop.read()
	s, fside = vidside.read()
	cv2.imshow("Bovenframe", ftop)
	cv2.imshow("Sideframe", fside)
	
	vingermasker = functies.contactVinger(fside)
	contact = functies.contactDubbel(sidemasker, vingermasker)
	if contact:
		(x, y) = functies.vingerPixel(ftop)
		letter = functies.welkeLetter(lijstKeyBox, x, y)	
		pbar.write("Gevonden letter: " + letter)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	pbar.update(1)
	time.sleep(0.1)	#zonder delay ongeveer 80/90fps

vidtop.release()
vidside.release()
cv2.destroyAllWindows()

