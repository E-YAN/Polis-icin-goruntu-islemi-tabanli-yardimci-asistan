
import face_recognition
import cv2
from imutils.video import VideoStream
import datetime
import argparse
import imutils
import time
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Load a sample picture and learn how to recognize it.
yahya_image = face_recognition.load_image_file("yahya.jpg")
yahya_face_encoding = face_recognition.face_encodings(yahya_image)[0]

# Load a second sample picture and learn how to recognize it.
najib_image = face_recognition.load_image_file("najib.jpg")
najib_face_encoding = face_recognition.face_encodings(najib_image)[0]

# Load a third sample picture and learn how to recognize it.
fuat_image = face_recognition.load_image_file("fuat.jpg")
fuat_face_encoding = face_recognition.face_encodings(fuat_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    yahya_face_encoding,
    najib_face_encoding,
    fuat_face_encoding
]
known_face_names = [
    "YAHYA",
    "NAJIB",
    "FUAT"
]
known_face_age = [
    "Age: 24", 
    "Age: 22",
    "Age: 25"
]
known_face_Id = [
    "Id: 1001", 
    "Id: 1357",
    "Id: 2468"
]
known_face_crime = [
    "CLEAN", 
    "MURDER",
    "CLEAN"
]


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
face_ages = []
face_Id = []
face_crime = []
process_this_frame = True

age = ""
Id = ""
crime = ""



ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
    help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())



vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    #frame = imutils.resize(frame)
    
    # Grab a single frame of video
    # frame = cv2.imread("testt.jpg", cv2.IMREAD_COLOR)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_ages = []
        face_Id = []
        face_crime = []
        
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                age = known_face_age[first_match_index]
                Id = known_face_Id[first_match_index]
                crime = known_face_crime[first_match_index]


            face_names.append(name)
            face_ages.append(age)
            face_Id.append(Id)
            face_crime.append(crime)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name, age, Id, crime in zip(face_locations, face_names, face_ages, face_Id, face_crime):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
    
        top -= 24
        right += 8
        bottom += 40
        left -= 8

        if name == "Unknown":
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom - 40), (0, 0, 0), 3)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 33), (right, bottom + 100), (0, 0, 0), 3) #cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 12, bottom + 39), font, 1.0, (0, 0, 0), 2)
        
        else:
            if crime == "CLEAN":
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom - 40), (0, 255, 0), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 34), (right, bottom + 100), (0, 255, 0), 2) #cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 12, bottom - 6), font, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, age, (left + 12, bottom + 26), font, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, Id, (left + 12, bottom + 56), font, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, crime, (left + 12, bottom + 86), font, 1.0, (0, 255, 0), 2)
            else:
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom - 40), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 34), (right, bottom + 100), (0, 0, 255), 2) #cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 12, bottom - 6), font, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, age, (left + 12, bottom + 26), font, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, Id, (left + 12, bottom + 56), font, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, crime, (left + 12, bottom + 86), font, 1.0, (0, 0, 255), 2)


    # Display the resulting image
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('result', 600,600)
    cv2.imshow('result', frame)
    
    key = cv2.waitKey(1)
    if key==27:
        break
    
cv2.waitKey(0)
cv2.destroyAllWindows()

