import numpy as np

check_list = [(1,2,3,0),(1,3,4,1),(2,4,5,2),(4,5,10,2),(7,2,1,0),(7,2,3,1)]
obj_true_box = np.arange(24).reshape(6,-1)
from_check_list_idx = 0
to_check_list_idx = 0
for i in range(8): # 8=bs->true_box.shape[0]
    target_img_idx = [x for x in check_list if x[0] == i]
    if len(target_img_idx) > 0:
        print(i)
        print('target_img_idx ', target_img_idx)
        to_check_list_idx = from_check_list_idx + len(target_img_idx)
        now_obj_true_box = obj_true_box[from_check_list_idx:to_check_list_idx]
        print('obj_true_box ', now_obj_true_box)
        from_check_list_idx = to_check_list_idx
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], x[1]), axis=-1),
        (pred_box[i], now_obj_true_box), tf.float32)
