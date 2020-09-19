import numpy as np

# example set ===
check_list = [(1,2,3,0),(1,3,4,1),(2,4,5,2),(4,5,10,2),(7,2,1,0),(7,2,3,1)]
obj_true_box = np.arange(24).reshape(6,-1)

# idx which is used when getting same img===
from_check_list_idx = 0
to_check_list_idx = 0

# iou ===
best_iou = tf.zeros((true_box.shape[0], grid_size, grid_size, tf.shape(anchor_idxs)[0], 1))  # [bs,grid,grid,anchors=3,iou=1]
indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
idx = 0

for i in range(true_box.shape[0]): # 8=bs->true_box.shape[0]
    target_imgs = [x for x in check_list if x[0] == i]  # [i,grid,grid,anchors]
    if len(target_imgs) > 0:
        print(i)
        print('target_imgs ', target_imgs)
        to_check_list_idx = from_check_list_idx + len(target_imgs)
        now_obj_true_box = obj_true_box[from_check_list_idx:to_check_list_idx]  # [xmin,ymin,xmax,ymax]
        print('obj_true_box ', now_obj_true_box)
        from_check_list_idx = to_check_list_idx
        for j in range(len(target_imgs)):
            now_img_info = target_imgs[j]  # [i,grid,grid,anchors]
            now_img_box = now_obj_true_box[j]  # [xmin,ymin,xmax,ymax]
            # calculate iou ===
            best_pred_anchor = pred_box[i][now_img_box[1]][now_img_box[2]][now_img_box[3]]  # [xmin,ymin,xmax,ymax]
            best_pred_anchor_wh = best_pred_anchor[0:2] - best_pred_anchor[2:4]  # [w,h]
            best_pred_anchor_area = best_pred_anchor_wh[0] * best_pred_anchor_wh[1]
            true_box_wh = now_img_box[0:2] - now_img_box[2:4]
            true_box_area = true_box_wh[0] * true_box_wh[1]
            intersection = tf.minimum(best_pred_anchor_wh[0], true_box_wh[0]) * \
                tf.minimum(best_pred_anchor_wh[1], true_box_wh[1])
            iou = intersection / (best_pred_anchor_area + true_box_area - intersection)

            indexes = indexes.write(idx, [i, now_img_info[1], now_img_info[2], now_img_info[3]]) # ある画像の、どのgridgridの、ベストアンカーのところ
            updates = updates.write(idx, [iou])  # 画像キーがiの画像のうちj番目の画像におけるbestanchorぽじのPBとTBのiou
            idx += 1

tf.tensor_scatter_nd_update(best_iou, indexes.stack(), updates.stack())  # sgapeエラー出たらbest_iouのanchoersの後の1を消してみる
ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)  # best anchorぽじのPB以外は0なのでok
