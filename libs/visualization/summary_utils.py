import tensorflow as tf


def visualize_input(gt_boxes, image, gt_masks):
    image_sum_sample = image[:1]
    visualize_masks(gt_masks, name="input_image_gt_mask")
    visualize_bb(image, gt_boxes)
    visualize_input_image(image_sum_sample)


def visualize_predictions(boxes, image, masks):
    image_sum_sample = image[:1]
    visualize_masks(masks, name="input_image_gt_mask")
    visualize_bb(image_sum_sample, boxes)

# TODO: Present all masks in different colors
def visualize_masks(masks, name):
    masks = tf.cast(masks, tf.float32)
    tf.summary.image(name=name, tensor=masks, max_outputs=1)


def visualize_bb(image, gt_boxes):
    image_sum_sample_shape = tf.shape(image)[1:]
    gt_x_min = gt_boxes[:, 0] / tf.cast(image_sum_sample_shape[1], tf.float32)
    gt_y_min = gt_boxes[:, 1] / tf.cast(image_sum_sample_shape[0], tf.float32)
    gt_x_max = gt_boxes[:, 2] / tf.cast(image_sum_sample_shape[1], tf.float32)
    gt_y_max = gt_boxes[:, 3] / tf.cast(image_sum_sample_shape[0], tf.float32)
    bb = tf.stack([gt_y_min, gt_x_min, gt_y_max, gt_x_max], axis=1)
    tf.summary.image(name="input_image_gt_bb",
                     tensor=tf.image.draw_bounding_boxes(image, tf.expand_dims(bb, 0), name=None),
                     max_outputs=1)

def visualize_input_image(image):
    tf.summary.image(name="input_image", tensor=image, max_outputs=1)
