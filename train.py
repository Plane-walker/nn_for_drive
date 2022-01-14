from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD
from import_data import load_train_data_hdf5, load_test_data_hdf5, data_preprocess, load_hdf5
from model import get_model
from tensorflow.keras.models import model_from_json
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def restore_patches(pred, N_imgs, img_h, img_w, stride_h, stride_w):
    N_channels = pred.shape[1]
    patch_h = pred.shape[2]
    patch_w = pred.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    prob = np.zeros((N_imgs,N_channels,img_h,img_w))
    N_overlap = np.zeros((N_imgs,N_channels,img_h,img_w))
    count = 0
    for i in range(N_imgs):
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                prob[i, :, (h*stride_h):((h*stride_h)+patch_h),
                (w*stride_w):((w*stride_w)+patch_w)] += pred[count]
                N_overlap[i, :, (h*stride_h):((h*stride_h)+patch_h),
                (w*stride_w):((w*stride_w)+patch_w)] += 1
                count += 1
    avg_pred = prob / N_overlap

    return avg_pred


def clean_outside(imgs, truth, masks):
    N_imgs = imgs.shape[0]
    img_h = imgs.shape[2]
    img_w = imgs.shape[3]
    y_pred = []
    y_true = []
    for i in range(N_imgs):
        for h in range(img_h):
            for w in range(img_w):
                if masks[i, :, h, w] != 0:
                    y_pred.append(imgs[i, :, h, w])
                    y_true.append(truth[i, :, h, w])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return y_pred, y_true


def train():
    patches = 1000
    patch_h = 48
    patch_w = 48
    patches_images_train, patches_masks_train = load_train_data_hdf5(patch_h, patch_w, patches)
    epochs = 5
    batch_size = 64
    lr = 0.1
    decay_rate = lr / epochs
    sgd = SGD(lr=lr, momentum=0.8, decay=decay_rate, nesterov=False)
    model = get_model(image_shape=(patch_h, patch_w, 1),
                      filters=32,
                      depth=4,
                      inc_rate=2,
                      activation='relu',
                      drop=0.2,
                      batch_norm=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    json_string = model.to_json()
    open('model_architecture.json', 'w').write(json_string)
    checkpoint = ModelCheckpoint(filepath='best_weights.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=20,
                                   verbose=1,
                                   mode='min')
    history = model.fit(patches_images_train,
                        patches_masks_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1,
                        callbacks=[checkpoint, early_stopping])


def test():
    patch_h = 48
    patch_w = 48
    stride_h = 47
    stride_w = 47
    test_images = load_hdf5('images_test.hdf5')
    test_images = data_preprocess(test_images)
    test_masks = load_hdf5('masks_test.hdf5')
    patches_images_test, new_h, new_w, test_truths = load_test_data_hdf5(patch_h, patch_w, stride_h, stride_w)
    with open(r'model_architecture.json', 'r') as file:
        model_json1 = file.read()
    model = model_from_json(model_json1)
    model.load_weights('best_weights.h5')
    predictions = model.predict(patches_images_test, batch_size=32, verbose=2)
    pred = np.empty((predictions.shape[0], 1, patch_h, patch_w))
    for i in range(predictions.shape[0]):
        for j in range(patch_h):
            for k in range(patch_w):
                pred[i, 0, j, k] = predictions[i, j, k, 1]
    N_imgs = test_images.shape[0]
    image_h = test_images.shape[2]
    image_w = test_images.shape[3]
    pred_images = restore_patches(pred, N_imgs, new_h, new_w, stride_h, stride_w)
    pred_images = pred_images[:, :, 0: image_h, 0: image_w]

    y_pred, y_true = clean_outside(pred_images, test_truths, test_masks)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    print(f'Area under the ROC curve: {auc}')
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("ROC_curve.png")
    y_true = y_true.astype(np.int16)
    for i in range(y_pred.shape[0]):
        if y_pred[i] < 0.5:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    cm = confusion_matrix(y_true, y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / sum(sum(cm))
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = 2 * cm[1, 1] / (2 * cm[1, 1] + cm[1, 0] + cm[0, 1])

    print(cm)
    print("accuracy: {:.4f}".format(accuracy))
    print("recall: {:.4f}".format(recall))
    print("precision: {:.4f}".format(precision))
    print("specificity: {:.4f}".format(specificity))
    print("F1: {:.4f}".format(f1))


if __name__ == '__main__':
    train()
    test()