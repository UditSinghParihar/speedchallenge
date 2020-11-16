from sys import exit, argv
import cv2 


if __name__ == '__main__':
	videoFile = argv[1]
	dirc = "/home/dhagash/udit/speedchallenge/data2/imagesTest/"

	images = []

	cap = cv2.VideoCapture(videoFile)
	if(cap.isOpened() == True):
		print("Successfully opened.")

	imgId = 0
	while(True):
		ret, frame = cap.read()
		
		if(ret):
			cv2.imwrite(dirc + "rgb{:06d}.jpg".format(imgId), frame)
			imgId += 1
		else:
			break

		# images.append(frame)

		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# cv2.imshow('Frame', frame)

		# if(cv2.waitKey(1) & 0xFF == ord('q')):
		# 	break

	cap.release()
	cv2.destroyAllWindows()