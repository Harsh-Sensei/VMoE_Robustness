import tensorflow as tf
import os

class Custom_Imagenet2012_val_tfds:

    def __init__(self, image_path="../data/ILSVRC2012_val/ILSVRC2012_img_val/",
                 label_path="../data/ILSVRC2012_val/ILSVRC2012_validation_ground_truth.txt",
                 batch_size=1024) -> None:
        print("Current working directory:", os.getcwd())
        self.image_path = image_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.imgname_class_dict = self._get_imgname_class_dict()

    
    def _get_imgname_class_dict(self):
        res_dict = {}
        with open(self.label_path) as f:
            for idx, line in enumerate(f):
                formatted_id = format(idx+1, '08d')
                image_name = f'ILSVRC2012_val_{formatted_id}.JPEG'
                res_dict[image_name] = int(str(line))
                if idx > 1023:
                    break
        
        return res_dict

    def _reading_image_file(self, image_name, label):
        image = tf.io.read_file(self.image_path + image_name)
        return {'image': image,'label': label}

    def get_tf_dataset(self):
        img_names = list(self.imgname_class_dict.keys())
        labels = list(self.imgname_class_dict.values())
        dataset = tf.data.Dataset.from_tensor_slices((img_names, labels))
        return dataset.map(self._reading_image_file)
    
    def get_tf_dataset_testing(self):
        img_names = list(self.imgname_class_dict.keys())
        labels = list(self.imgname_class_dict.values())
        dataset = tf.data.Dataset.from_tensor_slices((img_names, labels))
        return dataset.map(self._reading_image_file).batch(9)