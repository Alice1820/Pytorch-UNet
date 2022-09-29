import cv2
import os 
import torch
import torch.nn.functional as F
import time

import PIL.Image as Image
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.metric import *

class_names =  ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
    'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
    'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
    'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']

# class_names = ['bed', 'books', 'ceiling', 'chair', 'floor', 'furniture', 'objects', 'picture', 'sofa', 'table', 'tv', 'wall', 'window']

def ensure_dir(path):
    if not os.path.isdir(path):
        try:
            # sleeptime = random.randint(0, 3)
            time.sleep(1)
            os.makedirs(path)
        except:
            print('conflict !!!')

def get_random_colors(num_classes=13):
    colors = []
    for i in range(num_classes):
        colors.append((np.random.random((1,3)) * 255).tolist()[0])

    return colors

def print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc, class_names=None, show_no_back=False, no_print=False):
    n = iou.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i+1)
        else:
            cls = '%d %s' % (i+1, class_names[i])
        lines.append('%-8s\t%.3f%%' % (cls, iou[i] * 100))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:])
    if show_no_back:
        lines.append('----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100, 'mean_IU_no_back', mean_IoU_no_back*100,
                                                                                                                'freq_IoU', freq_IoU*100, 'mean_pixel_acc', mean_pixel_acc*100, 'pixel_acc',pixel_acc*100))
    else:
        lines.append('----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100, 'freq_IoU', freq_IoU*100, 
                                                                                                    'mean_pixel_acc', mean_pixel_acc*100, 'pixel_acc',pixel_acc*100))
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line

def get_class_colors(num_classes=13):
    def uint82bin(n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

    N = num_classes
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    class_colors = cmap.tolist()
    return class_colors

def evaluate(net, dataloader, criterion, device, args):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    val_score = 0
    # iterate over the validation set
    hist = np.zeros((args.num_classes, args.num_classes))
    correct = 0
    labeled = 0
    count = 0

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        name = batch['name'][0]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long) # [1, 13, 480, 640]
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                one_hot = F.one_hot(mask_true, 256)[..., :net.n_classes].contiguous().permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                # dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=True)
                val_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], one_hot[:, 1:, ...], reduce_batch_first=False)
            # val_score += criterion(mask_pred, mask_true)
        mask_pred = F.softmax(mask_pred, dim=1).max(dim=1)[1]
        # print (mask_pred.shape) # [1, 480, 640]
        mask_pred = mask_pred.squeeze().data.cpu().numpy()
        mask_true = mask_true.squeeze().data.cpu().numpy()
        hist_tmp, labeled_tmp, correct_tmp = hist_info(args.num_classes, mask_pred, mask_true)
        # results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}
        
        hist += hist_tmp
        correct += correct_tmp
        labeled += labeled_tmp
        count += 1

        if args.save_path is not None:
            ensure_dir(args.save_path)
            ensure_dir(args.save_path+'_color')

            fn = name + '.png'

            # save colored result
            result_img = Image.fromarray(mask_pred.astype(np.uint8), mode='P')
            class_colors = get_class_colors(args.num_classes)
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(args.save_path+'_color', fn))

            # save raw result
            cv2.imwrite(os.path.join(args.save_path, fn), mask_pred)

    iou, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, labeled, correct)
    # Fixes a potential division by zero error
    # if num_val_batches == 0:
    #     return dice_score
    # return dice_score / num_val_batches
    result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc, class_names, show_no_back=False)
    return val_score / num_val_batches, result_line
