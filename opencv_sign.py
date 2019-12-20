import numpy as np
import cv2
from keras.models import load_model


model = load_model('Load your model')


classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


def predict(model,image):
    
    data = image.reshape(1,32,32,3)
    prediction_prob = model.predict(data)[0]

    predicted_class = list(prediction_prob).index(max(prediction_prob))
    p = classes[predicted_class]
    return max(prediction_prob),p


def crop_image(image,x,y,width,height):
    return image[y:y+height,x:x+width]

cap = cv2.VideoCapture(0)

x,y,height,width = 20,100,300,300
def main():
    while True:
   
       
        ret, frame = cap.read()

        cropped_img =frame[y:y+height,x:x+width]
        cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2)
        blurred_img = cv2.GaussianBlur(cropped_img,(15,15),0)
    
        resized_img = cv2.resize(blurred_img,(32,32),interpolation = cv2.INTER_AREA)
    # our_image = process_image(resized_img)
        # filename = 'checkimg.jpg'
        # my_image = cv2.imwrite(filename,resized_img)
        
        prediction_prob, predicted_class = predict(model,resized_img)
        # print(predicted_class,prediction_prob)

        # cv2.imshow("Cropped image", cropped_img)
        # cv2.imshow("Resized Image",resized_img)
        # cv2.imshow("Blurred image",blurred_img)
        cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
        cv2.putText(frame,predicted_class,(x,y-10), 1,2,(255,0,0),2)
        cv2.imshow('Video',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release() 
    cv2.destroyAllWindows()

    print(prediction_prob, predicted_class)
   
        

if __name__ =="__main__":
    main()



