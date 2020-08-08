from facenet_pytorch import MTCNN
import time
import torch
import torchvision
import matplotlib.pyplot as plt


def draw_box(image, faces, labels, class_names):
    plt.imshow(image)
    
    ax = plt.gca()  # get the context for drawing boxes

    params = {'font.size': 10, 'font.weight': 'bold'}
    plt.rcParams.update(params)

    # plot all boxes
    for i in range(len(faces)):
        x, y, x_right, y_bottom = faces[i].round().astype('int32')

        color = 'green' if labels[i] else 'red'

        params = {'text.color': color}
        plt.rcParams.update(params)
 
        rect = plt.Rectangle((x, y), x_right-x, y_bottom-y, fill=False, color=color, lw=2)
        
        ax.add_patch(rect)
        plt.text(x, y-15, class_names[labels[i]])
        
    plt.show()


def complete_model(image_name):
    
    detector = MTCNN(min_face_size=60)
    model = torch.load('models/resnet_normalization_10epochs_acc0993392.pt').cuda()

    start_time = time.time()

    image = plt.imread(image_name)

    time_elapsed = time.time() - start_time
    print(f'Time spent: {time_elapsed}')
    start_time = time.time()
    
    faces, _ = detector.detect(image)

    time_elapsed = time.time() - start_time
    print(f'Time spent: {time_elapsed}')
    start_time = time.time()
    
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), 
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    to_image = torchvision.transforms.ToPILImage()

    batch_images = []

    for face in faces:
        bounding_box = face
        x_left, y_top, x_right, y_bottom = bounding_box.round().astype('int32')

        cropped_face = image[y_top:y_bottom, x_left:x_right]
        
        cropped_face = transform(to_image(cropped_face))

        batch_images.append(cropped_face)

    batch_torch = torch.stack(batch_images).cuda()
    output = model(batch_torch)
    output = torch.log_softmax(output, -1)
    _, predicted = torch.max(output, 1)

    time_elapsed = time.time() - start_time
    print(f'Time spent: {time_elapsed}')

    draw_box(image, faces, predicted, ['WITHOUT MASK', 'WITH MASK'])


# name = input('Enter name:')
complete_model('image3.jpg')