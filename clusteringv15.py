from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os, re, torch
from sklearn.decomposition import PCA
import itertools, glob


def crop_detected_face(img, boxes, show_image=True, print_box=False):
    # todo: generalize for more faces. now its always overwritten
    for box in boxes:
        cropped_img = img.crop(box)
        if show_image:
            cropped_img.show()

    if print_box:
        frame_draw = img.copy()
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=1)

    return cropped_img


def add_noise(img_piece, sigma=25, blending_factor=0.5, show_image=False):
    img_noise = Image.effect_noise(img_piece.size, sigma).convert('RGB')

    img_with_noise = Image.blend(img_piece, img_noise, blending_factor)
    if show_image:
        img_with_noise.show()

    return img_with_noise

def add_noise_single(img, landmark, sigma=25, show_image=False):
    img_noise = Image.effect_noise((landmark[2] - landmark[0], landmark[3] - landmark[1]), sigma).convert('RGB')
    img_with_noise = img.copy()

    img_with_noise.paste(img_noise, box=landmark, mask=None)
    if show_image:
        img_with_noise.show()

    return img_with_noise

def add_noise_levels(img, landmark, save_path=''):
    noise_factors = np.arange(0,1,0.2)
    for noise_factor in noise_factors:
        img_with_noise = add_noise_single(img, landmark)
        Image.blend(img,img_with_noise, noise_factor).save((save_path + '/noise_test_' + str(noise_factor) + '.jpg'))
    return

def get_feature_bboxes(img, boxes, landmarks,save_path=''):
    """
    calculates the bounding box coordinates for eyes, nose and mouth scaled down to a 160x160 picture
    boxes: numpy.ndarray as a return value of the mtcnn.detect() function
    landmarks: numpy.ndarray as a return value of the mtcnn.detect() function
    """
    #clear savepath
    print("Clear landmarks_savepath.")
    data = list(itertools.chain(*(glob.glob(save_path + '*.%s' % ext) for ext in ["jpg", "jpeg", "png"])))
    for f in data:
        os.remove(f)
    
    left_eye = (landmarks[0][0][0], landmarks[0][0][1])
    right_eye = (landmarks[0][1][0], landmarks[0][1][1])
    nose = (landmarks[0][2][0], landmarks[0][2][1])
    mouth_left = (landmarks[0][3][0], landmarks[0][3][1])
    mouth_right = (landmarks[0][4][0], landmarks[0][4][1])

    top_left = (boxes[0][0], boxes[0][1])
    top_right = (boxes[0][2], boxes[0][1])
    bottom_left = (boxes[0][0], boxes[0][3])
    bottom_right = (boxes[0][2], boxes[0][3])

    box_width = top_right[0] - top_left[0]
    box_height = bottom_left[1] - top_left[1]

    left_eye_160 = (((left_eye[0]-top_left[0])/box_width)*160, ((left_eye[1]-top_left[1])/box_height)*160)
    right_eye_160 = (((right_eye[0]-top_left[0])/box_width)*160, ((right_eye[1]-top_left[1])/box_height)*160)
    nose_160 = (((nose[0]-top_left[0])/box_width)*160, ((nose[1]-top_left[1])/box_height)*160)
    mouth_left_160 = (((mouth_left[0]-top_left[0])/box_width)*160, ((mouth_left[1]-top_left[1])/box_height)*160)
    mouth_right_160 = (((mouth_right[0]-top_left[0])/box_width)*160, ((mouth_right[1]-top_left[1])/box_height)*160)

    eyes_y = (left_eye_160[1] + right_eye_160[1]) / 2
    mouth_y = (mouth_right_160[1] + mouth_left_160[1]) / 2

    eyes = (int(left_eye_160[0] - 20), int(left_eye_160[1] - 8), int(right_eye_160[0] + 20), int(right_eye_160[1] + 8))
    nose = (int((mouth_left_160[0] + nose_160[0]) / 2), int(eyes_y), int((mouth_right_160[0] + nose_160[0]) / 2), int((nose_160[1] + mouth_y) / 2))
    mouth = (int(mouth_left_160[0] - 5), int((nose_160[1] + mouth_y) / 2), int(mouth_right_160[0] + 5), int(mouth_y + (mouth_y-(nose_160[1] + mouth_y) / 2)))
    
    boxes ={}
    boxes['eyes'] = eyes
    boxes['nose'] = nose
    boxes['mouth'] = mouth
    for key in boxes:
        chop = img.crop(boxes[key])  
        chop.save(save_path + key +'.jpg')
    
    # returns are expected to be (x_min, y_min, x_max, y_max) for each landmark
    return eyes, nose, mouth


def crop_feature(img, feature):
    return img.crop(feature)


def get_noisy_features(img, feature, feature_bbox, noise_factors):
    noisy_pics = []

    new_img = img

    for noise in noise_factors:
        noisy_feature = add_noise(feature, blending_factor=noise)
        new_img.paste(noisy_feature, feature_bbox)
        # new_img.show()

        noisy_pics.append(new_img)
        new_img = img

    return noisy_pics


########################################################################################################################################################
def cluster_face(img, denominator=5, save_path=''):
    #clear savepath
    print("Clear boxes_savepath.")
    data = list(itertools.chain(*(glob.glob(save_path + '*.%s' % ext) for ext in ["jpg", "jpeg", "png"])))
    for f in data:
        os.remove(f)
        
    width, height = img.size

    chopsize = int(width / denominator)

    cropped_pieces = []

    # Save Chops of original image
    for y in range(0, width, chopsize):
        for x in range(0, height, chopsize):
            box = (x, y,
                   x + chopsize if x + chopsize < width else width - 1,
                   y + chopsize if y + chopsize < height else height - 1)
            #print(box)
            chop = img.crop(box)
            
            ys = str(y).zfill(4)
            xs = str(x).zfill(4)
            chop.save((save_path +'y'+ys+'x'+xs+'.jpg'))

            cropped_pieces.append(box)

    return cropped_pieces

def cluster_embeddings(img, noise_factors,chop_path,chops,mtcnn,resnet,save_path=''):
    new_comps={}
    
    # clear savepath
    print("Clear noise_savepath.")
    data = list(itertools.chain(*(glob.glob(save_path + '*.%s' % ext) for ext in ["jpg", "jpeg", "png"])))
    #print(data)
    for f in data:
        os.remove(f)
    chop_data = []
    for picture in os.listdir(chop_path):
        chop_data.append(re.match('.*jpg|.*jpeg|.*png$', picture).string)

    # generate images with noise
    print("Generating cluster embeddings",end="")
    j=0
    for chop in chop_data:
        new_tensors = []
        new_img = img.copy()
        chop_img = Image.open(chop_path + chop)
        for noise in noise_factors:
            noise_chop = add_noise(chop_img,blending_factor=noise)
            new_img.paste(noise_chop, (chops[j][0],chops[j][1]))
            new_img.save(save_path + chop_data[j] + "_" + str(noise) + ".jpg")

            #generate embedding for image
            print('.',end='')
            new_img_cropped = mtcnn(new_img)
            new_img_embedding = resnet(new_img_cropped.unsqueeze(0))
            new_tensors.append(new_img_embedding)
        # merge noise tensors into numpy array
        tensor_array = torch.cat((new_tensors), 0)
        np_array = tensor_array.detach().numpy()
        j = j + 1
        # Update the dictionary: Choose a name for your component and save the numpy-array into it
        comp = 'A'+str(j)
        new_comps[comp] = np_array

    print("Finished!")
    return new_comps


def create_dict_with_embeddings(path, database, resnet):
    print("\nEmbeddings are being calculated", end="")

    name_dict = {}

    for person in database:
        tensor_list = []
        images = os.listdir(path + person)
        for image in images:
            # Open Image in person-folder
            db_img = Image.open(path + person + '/' + image)

            # Get cropped and prewhitened image tensor
            db_img_cropped = mtcnn(db_img)

            # Calculate embedding (unsqueeze to add batch dimension)
            db_img_embedding = resnet(db_img_cropped.unsqueeze(0))

            # Save embeddings in 'tensor_list'
            tensor_list.append(db_img_embedding)
        # Save tensor_list for a person in 'name_dict'
        name_dict[person] = tensor_list

        print(".", end="")

    print("\n\nFinished!")

    return name_dict


def plot_pca(comp_dict):
    score = []
    pca = PCA()
    t = 0
    for comp in comp_dict:
        X = comp_dict[comp]

        pca.fit(X)
        pca_data = pca.transform(X)

        per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
        labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

        plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.title(comp)
        plt.show()
        i=0
        dim=0
        sco=0
        while dim < 0.95:
            dim += pca.explained_variance_ratio_[i]
            sco += pca.singular_values_[i] 
            i+=1
        score.append(sco)
        t+=1
        #print('A' + str(t) +': ' + str(score[t-1]))
    return score


def plot_heat_map(chops, score,img_path='',cmap='inferno',save_path='',save_name=''):
    pic = np.zeros((160, 160))
    i = 0
    for clust in chops:
        x = clust[0]
        while (x - 1) < clust[2]:
            y = clust[1]
            while (y - 1) < clust[3]:
                pic[y, x] = score[i]
                y += 1
            x += 1
        i += 1
    cnorm = colors.Normalize(vmin=0, vmax=1)
    org_img = plt.imshow(Image.open(img_path))
    heatmap = plt.imshow(pic, cmap,norm=cnorm, interpolation='nearest', alpha=0.6)
    #plt.legend()
    plt.savefig(save_path + save_name + '.jpg')
    #plt.show()
    return pic


########################################################################################################################################################
if __name__ == '__main__':
    file_path = 'cropped_img.png'
    img = Image.open(file_path)

    mtcnn = MTCNN()
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

    # cropped_face = crop_detected_face(img, boxes)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img)
    ax.axis('off')
    fig.show()

    eyes_bbox, nose_bbox, mouth_bbox = get_feature_bboxes(landmarks)

    cropped_eyes = crop_feature(img, eyes_bbox)
    # cropped_nose = crop_feature(img, nose_bbox)
    # cropped_mout = corp_feature(img, mouth_bbox)

    noise_factors = np.arange(0, 1, 0.1)

    pics_eyes_noisy = get_noisy_features(img, cropped_eyes, eyes_bbox, noise_factors)

    for pic_num in range(len(pics_eyes_noisy)):
        pics_eyes_noisy[pic_num].show()
