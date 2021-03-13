from PIL import Image

def load_tf_bert_weights_to_torch(model, ckpt_path):
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    abs_ckpt_path = os.path.abspath(ckpt_path)
    init_vars = tf.train.list_variables(abs_ckpt_path)
    names = []
    weights = []
    for name, shape in init_vars:
        print("Loading TF tensor {} with shape {}".format(name, shape))
        weight = tf.train.load_variable(abs_ckpt_path, name)
        names.append(name)
        weights.append(weight)

    for name, weight in zip(names, weights):
        name = name.split('/')
        if any(n in ["adam_v", "adam_m" "global_step"] for n in name):
            print("Skipping TF tensor {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            weight = np.transpose(weight)
        try:
            assert pointer.shape == weight.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model

def cached_path(url_or_filename, cache_dir=None):
    pass

# transform = transforms.Compose([
#     transforms.RandomCrop(224),  # args.crop_size, by default it is set to be 224
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406),
#                             (0.229, 0.224, 0.225))])

def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image