import keras
from utils.eval import evaluate
from utils.coco_eval import evaluate_coco


class Evaluate(keras.callbacks.Callback):
    """
    Evaluation callback for arbitrary datasets.
    """

    def __init__(
            self,
            generator,
            iou_threshold=0.5,
            score_threshold=0.05,
            max_detections=100,
            save_path=None,
            tensorboard=None,
            weighted_average=False,
            verbose=1
    ):
        """
        Evaluate a given dataset using a given model at the end of every epoch during training.

        Args:
            generator: The generator that represents the dataset to evaluate.
            iou_threshold: The threshold used to consider when a detection is positive or negative.
            score_threshold: The score confidence threshold to use for detections.
            max_detections: The maximum number of detections to use per image.
            save_path: The path to save images with visualized detections to.
            tensorboard: Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average: Compute the mAP using the weighted average of precisions among classes.
            verbose: Set the verbosity level, by default this is set to 1.
        """
        self.generator = generator
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.weighted_average = weighted_average
        self.verbose = verbose

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path,
            epoch=epoch
        )

        # compute per class average precision
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.mean_ap
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))


class RedirectModel(keras.callbacks.Callback):
    """
    Callback which wraps another callback, but executed on a different model.

    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```

    Args
        callback : callback to wrap.
        model : model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)


class CocoEval(keras.callbacks.Callback):
    """ Performs COCO evaluation on each epoch.
    """
    def __init__(self, generator, tensorboard=None, threshold=0.05):
        """ CocoEval callback intializer.

        Args
            generator   : The generator used for creating validation data.
            tensorboard : If given, the results will be written to tensorboard.
            threshold   : The score threshold to use.
        """
        self.generator = generator
        self.threshold = threshold
        self.tensorboard = tensorboard

        super(CocoEval, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        coco_tag = ['AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50      | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.75      | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]']
        coco_eval_stats = evaluate_coco(self.generator, self.model, self.threshold)
        if coco_eval_stats is not None and self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            for index, result in enumerate(coco_eval_stats):
                summary_value = summary.value.add()
                summary_value.simple_value = result
                summary_value.tag = '{}. {}'.format(index + 1, coco_tag[index])
                self.tensorboard.writer.add_summary(summary, epoch)
                logs[coco_tag[index]] = result
