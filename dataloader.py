from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator


class DataLoader:
    num_classes = 2

    def __init__(self, dataset_path, image_folder_name, mask_folder_name, seed=23333):
        self.dataset_path = dataset_path
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.seed = seed

    def generate(self, batch_size, target_size):
        aug_dict = {}

        image_data_generator = ImageDataGenerator(**aug_dict, rescale=1 / 255)
        mask_data_generator = ImageDataGenerator(**aug_dict, rescale=1 / 38)

        image_generator = image_data_generator.flow_from_directory(
            self.dataset_path,
            classes=[self.image_folder_name],
            class_mode=None,
            color_mode="grayscale",
            target_size=target_size,
            batch_size=batch_size,
            seed=self.seed,
        )

        mask_generator = mask_data_generator.flow_from_directory(
            self.dataset_path,
            classes=[self.mask_folder_name],
            class_mode=None,
            color_mode="grayscale",
            target_size=target_size,
            batch_size=batch_size,
            seed=self.seed,
        )

        # for img, mask in zip(image_generator, mask_generator):
        #     mask = to_categorical(mask, self.num_classes)
        #     assert mask.shape[-1] == self.num_classes
        #     yield img, mask

        return image_generator
