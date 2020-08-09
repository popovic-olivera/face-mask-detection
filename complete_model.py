from facenet_pytorch import MTCNN
import time
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt


detector = MTCNN()
model = torch.load('models/mobilenet_LRDL_80_20_epochs20_acc0992274.pt', map_location=torch.device('cpu'))


def draw_box(image, resized, faces, labels, class_names):
    im_height, im_width, _ = image.shape
    res_height, res_width, _ = resized.shape

    # plot all boxes
    for i in range(len(faces)):
        x, y, x_right, y_bottom = faces[i].round().astype('int32')

        x = round(im_width * x / res_width)
        y = round(im_height * y / res_height)
        x_right = round(im_width * x_right / res_width)
        y_bottom = round(im_height * y_bottom / res_height)

        color = (0, 255, 0) if labels[i] else (0, 0, 255)

        cv2.putText(image, class_names[labels[i]], (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (x, y), (x_right, y_bottom), color, 2)
        
    return image


def complete_model(image):
    start_time = time.time()

    height, width, _ = image.shape

    scale_percent = 60
    width = max(int(width * scale_percent / 100), 224)
    height = max(int(height * scale_percent / 100), 224)
    resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

    faces, _ = detector.detect(resized)

    if faces is None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    time_elapsed = time.time() - start_time
    print(f'Time spent: {time_elapsed}')
    start_time = time.time()
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    batch_images = []

    for face in faces:
        startX, startY, endX, endY = face.round().astype('int32')
        
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(width - 1, endX), min(height - 1, endY))

        cropped_face = resized[startY:endY, startX:endX]
        cropped_face = cv2.resize(cropped_face, (224, 224))

        cropped_face = transform(cropped_face)

        batch_images.append(cropped_face)

    batch_torch = torch.stack(batch_images)
    output = model(batch_torch)
    _, predicted = torch.max(output, 1)

    time_elapsed = time.time() - start_time
    print(f'Time spent: {time_elapsed}')

    return draw_box(image, resized, faces, predicted, ['WITHOUT MASK', 'WITH MASK'])


def on_image(image_name):
    image = plt.imread(image_name)
    cv2.imshow('Model output', complete_model(image))


def on_video(video_name):
    cap = cv2.VideoCapture(video_name)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))

    # read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            new_frame = complete_model(frame)
            out.write(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))

            # Display the resulting frame
            # cv2.imshow('Output', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else: 
            break


    # When everything done, release the video capture object
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
