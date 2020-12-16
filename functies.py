import cv2
import numpy as np
from tqdm import tqdm

def neemBackground(pathvid, frame, pathimg): #sla frame met index 'frame' op als afbeelding
	vid = cv2.VideoCapture(pathvid)
	vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
	s, f = vid.read()
	cv2.imwrite(pathimg, f)
	vid.release()
	return f

def draai180(pathin, pathout): #draai video 180°

	vid = cv2.VideoCapture(pathin)

	w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(vid.get(cv2.CAP_PROP_FPS))
	frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	out = cv2.VideoWriter(pathout, fourcc, fps, (w,h))

	matrix = cv2.getRotationMatrix2D((w/2, h/2), 180, 1) #rotatie rond middenpunt, 180°, schaal 1:1

	print("Draaien van '"+pathin+"' 180° in '"+pathout+"'")
	for i in tqdm(range(frames)):
		s, framein = vid.read()
		frame = cv2.warpAffine(framein, matrix, (w,h))
		out.write(frame)
	vid.release()
	out.release()

def maakToetsBoxFile(pathtxt, pathpng):	#bepaal grenzen van elke toets/letter op basis van user input en bewaar coordinaten in file
	
	letters = list(range(ord('A'), ord('Z')+1)) #lijst met ascii waardes voor alle letters
	lbx = [] #lijsten om x en y coord van lb en ro hoeken bij te houden
	lby = []
	rox = []
	roy = []
	l = 0
	img = cv2.imread(pathpng)

	###callback start
	def klikPixel(event, x, y, flags, param):
		nonlocal l, letters, lbx, lby, rox, roy #nonlocal scope = this and parent function
		if l >= len(letters): #voorkomen dat er te veel bounding boxes gemaakt worden
			pass
		if event == cv2.EVENT_LBUTTONDOWN:
			lbx.append(int(x))
			lby.append(int(y))
		if event == cv2.EVENT_LBUTTONUP:
			rox.append(int(x))
			roy.append(int(y))
			cv2.rectangle(img, (lbx[l], lby[l]), (rox[l], roy[l]), (0, 255, 0), 1)
			cv2.imshow("MaakToetsBoxFile", img)
			l += 1
			if l < len(letters):
				print("Teken bounding box rond toets "+ chr(letters[l]))
			else:
				print("Alle toetsen zijn aangeduid, druk op een toets om af te sluiten en file te genereren")
	###callback end

	cv2.namedWindow("MaakToetsBoxFile")
	cv2.imshow("MaakToetsBoxFile", img)
	cv2.setMouseCallback("MaakToetsBoxFile", klikPixel)
	print("Teken bounding box rond toets "+ chr(letters[l])) #eerste keer, daarna in callback
	
	while(l < len(letters)): #wachten tot alle letters ingegeven zijn, vervolgens keypress om verder te gaan
	        cv2.waitKey()
	cv2.destroyWindow("MaakToetsBoxFile")

	file = open(pathtxt, "w")
	for i in range(len(letters)):
	        lijn = "%s;%d;%d;%d;%d;\n" %(chr(letters[i]), lbx[i], lby[i], rox[i], roy[i])
	        file.write(lijn)
	file.close()


def maakToetsGrensFile(pathtxt, pathpng): #bepaal grenzen van toetsenbord op basis van user input en bewaar coordinaten in file
	lx = [] #lijsten om x en y coord van lb en ro hoeken bij te houden
	ly = []
	img = cv2.imread(pathpng)
	###callback start
	def klikPixel(event, x, y, flags, param):
		nonlocal lx, ly
		if event == cv2.EVENT_LBUTTONDOWN:
			lx.append(int(x))
			ly.append(int(y))
	###callback end

	cv2.namedWindow("MaakToetsBoxFile")
	cv2.imshow("MaakToetsBoxFile", img)
	cv2.setMouseCallback("MaakToetsBoxFile", klikPixel)
	print("Teken veelhoek rond de contour van het toetsenbord")
	print("Druk op een toets om af te sluiten")
	
	while True:
		if cv2.waitKey():
			break
	cv2.destroyWindow("MaakToetsBoxFile")

	file = open(pathtxt, "w")
	for i in range(len(lx)):
		lijn = "%d;%d;\n" %(lx[i], ly[i])
		file.write(lijn)
	file.close()

def leesKeyboundIn(path): #lees bounding box van elke toets/letter in op basis van file
        lijst = []
        file = open(path, "r")
        for lijn in file:
                (letter, linksbovenx, linksboveny, rechtsonderx, rechtsondery, _) = lijn.split(";")
                data = (letter, (int(linksbovenx), int(linksboveny)), (int(rechtsonderx), int(rechtsondery)))
                lijst.append(data)
        file.close()
        return lijst

def leesToetsgrensIn(path): #lees genscoordinaten van toetsenbord in op basis van file
	lijst = []
	file = open(path, "r")
	for lijn in file:
		(x, y, _) = lijn.split(";")
		data = (int(x), int(y))
		lijst.append(data)
	file.close()
	pts = np.array(lijst, np.int32)
	pts= pts.reshape((-1, 1, 2))
	return pts

def vingerPixel(img):	#geef coordinaat van vingertop (groene stip) in afbeelding
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	seglow = np.array([20, 10, 80])
	seghigh = np.array([100, 230, 220])
	mask = cv2.inRange(hsv, seglow, seghigh) #hsv segmentatie van groene stip op vingertop
	
	kernel = np.ones((7,7), np.uint8)
	erodil = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	contours, hierarchy = cv2.findContours(erodil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	if len(contours) == 0:	#geen contour gevonden (dus geen vingertop gevonden)
		return (0, 0)
	
	m = max(contours, key = cv2.contourArea)  #zoek grootste contour volgens oppervlakte (ruis wegwerken)
	(x,y), r = cv2.minEnclosingCircle(m)	#neem middenpunt van gebied (= midden van groene stip op vinger)
	
	#debug info
	cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), 2)
	cv2.imshow("ToetsPunt", img)
	
	return (x,y)

def welkeLetter(boxes, x, y):	#zet coordinaten om naar letter op basis van bouding boxes van elke letter
	for (letter, (x1, y1), (x2, y2)) in boxes:
		if (x1 <= x <= x2) and (y1 <= y <= y2):
			return letter
	return ""

def contactSide(img, pts): #maak masker van polygon (contour van het toetsenbord)
	mask = np.zeros(img.shape, dtype=np.uint8)
	mask = cv2.fillPoly(mask,[pts],(0,255,255)) #niet wit voor duidelijkere visualisatie in contactDubbel
	return mask

def contactVinger(img):	#vind vinger op basis van hsv segmentatie en maak masker
	masker = np.zeros(img.shape, dtype=np.uint8)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	seglow = np.array([0, 40, 0])
	seghigh = np.array([50, 255, 255])
	mask = cv2.inRange(hsv, seglow, seghigh)
	kernel = np.ones((20,20), np.uint8)
	filter = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	contours, hierarchy = cv2.findContours(filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	cv2.drawContours(masker, contours, -1, (255, 255, 0), -1) #niet wit voor duidelijkere visualisatie in contactDubbel
	return masker 

def contactDubbel(toetsenbord, vinger): #return True indien overlap tussen de twee maskers (vinger raakt toetsenbord)
	overlapvis = cv2.bitwise_or(toetsenbord, vinger)
	cv2.imshow("OverlapMask", overlapvis)	#visualisatie overlap
	overlap = cv2.bitwise_and(toetsenbord, vinger)
	over = cv2.cvtColor(overlap, cv2.COLOR_BGR2GRAY)
	cont, h = cv2.findContours(over, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return len(cont) >= 1

