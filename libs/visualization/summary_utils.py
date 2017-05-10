import tensorflow as tf


def visualize_input(boxes, image, masks):
    image_sum_sample = image[:1]
    visualize_masks(masks, "input_image_gt_mask")
    visualize_bb(image, boxes, "input_image_gt_bb")
    visualize_input_image(image_sum_sample)


def visualize_rpn_predictions(boxes, image):
    image_sum_sample = image[:1]
    visualize_bb(image_sum_sample, boxes, "rpn_pred_bb")

# TODO: Present all masks in different colors
def visualize_masks(masks, name):
    masks = tf.cast(masks, tf.float32)
    tf.summary.image(name=name, tensor=masks, max_outputs=1)


def visualize_bb(image, boxes, name):
    image_sum_sample_shape = tf.shape(image)[1:]
    gt_x_min = boxes[:, 0] / tf.cast(image_sum_sample_shape[1], tf.float32)
    gt_y_min = boxes[:, 1] / tf.cast(image_sum_sample_shape[0], tf.float32)
    gt_x_max = boxes[:, 2] / tf.cast(image_sum_sample_shape[1], tf.float32)
    gt_y_max = boxes[:, 3] / tf.cast(image_sum_sample_shape[0], tf.float32)
    bb = tf.stack([gt_y_min, gt_x_min, gt_y_max, gt_x_max], axis=1)
    tf.summary.image(name=name,
                     tensor=tf.image.draw_bounding_boxes(image, tf.expand_dims(bb, 0), name=None),
                     max_outputs=1)


def visualize_input_image(image):
    tf.summary.image(name="input_image", tensor=image, max_outputs=1)


def visualize_final_predictions(boxes, image, masks):
    visualize_masks(masks, "pred_mask")
    visualize_bb(image, boxes, "final_bb_pred")
