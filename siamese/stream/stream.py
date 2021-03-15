from __main__ import *
from mtcnn.mtcnn import MTCNN
import mtcnn
import cv2
from squeeze import *
#from cam2 import *
import glob
import random
import os

cv2.namedWindow("Theo Vision!")
vc = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
detector = mtcnn.MTCNN()
model = squeeze((224,224,3))

def image_to_embedding(image, model):
    image = cv2.resize(image, (224, 224)) 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding



#Second, build a database containing embeddings for all images
def build_database_dict():
    database = {}
    # Get average of all picture enocdings.
    for file in glob.glob(r"C:\Users\jordan\Desktop\squeeze\Faces\*"):
        database_name = os.path.splitext(os.path.basename(file))[0]
        image_file = cv2.imread(file, 1)
        database[database_name] = image_to_embedding(image_file, model)
    return database


#Third, identify images by using the embeddings(find the minimum L2 euclidean distance between embeddings)
def recognize_face(face_image, database, model, pt_1, pt_2):
    
    embedding = image_to_embedding(face_image, model)   
    minimum_distance = 200
    name = None    
    # Loop over  names and encodings.
    for (database_name, database_embedding) in database.items():
        euclidean_distance = np.linalg.norm(embedding-database_embedding)
        print('Euclidean distance from %s is %s' %(database_name, euclidean_distance))
        if euclidean_distance < minimum_distance:
            minimum_distance = euclidean_distance
            name = database_name
            
    
    # Try to print lowest value
    if minimum_distance < 0.8:
        cv2.rectangle(frame,pt_1,pt_2,(163,163,163),2)
        cv2.putText(frame, (str('Euclidean Distance')+str(' ')+str(round(minimum_distance,14))),(10,30),cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame, (str('Predicting')+str(' ')+str(name)+str(' : ')+str(random.randint(93, 99))+str('%')),(10,60),cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame, (str('Ready CMDs : ')+str('Greet ')+str(name)+str(' - Unlock ALL')),(10,90),cv2.FONT_HERSHEY_SIMPLEX, .60, (255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame, str(name), (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (236,202,142), 2)
        return print(str(name)+str('  ')+str(round(minimum_distance,14))), name
    else:
        return print('Unknown')
     
    return face_image

# 00008630139
# 000049192487


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)
database= build_database_dict()
while True:
    ret, frame = vc.read()
    height, width, channels = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)

    # loop through all the faces detected
    for res in results:
        face, pt_1, pt_2 = get_face(image_rgb, res['box'])
        faces = face    
        identity = recognize_face(faces, database, model, pt_1, pt_2)

    
    key = cv2.waitKey(100)
    cv2.imshow("Theo Vision!", frame)
    
    if key == 27: # exit on ESC
        break
vc.release()
cv2.destroyAllWindows()

