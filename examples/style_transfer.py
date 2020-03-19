import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import dezero
import dezero.functions as F
from dezero import Variable
from dezero.models import VGG16


use_gpu = dezero.cuda.gpu_enable
lr = 5.0
iterations = 2001
model_input_size = (224, 224)
style_weight = 1.0
content_weight = 1e-4
total_varitaion_weight = 1e-6
content_layers = ['conv5_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
content_url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
style_url = 'https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/starry_night_google.jpg'


class VGG16(VGG16):
    def extract(self, x):
        c1_1 = F.relu(self.conv1_1(x))
        c1_2 = F.relu(self.conv1_2(c1_1))
        p1 = F.average_pooling(c1_2, 2, 2)
        c2_1 = F.relu(self.conv2_1(p1))
        c2_2 = F.relu(self.conv2_2(c2_1))
        p2 = F.average_pooling(c2_2, 2, 2)
        c3_1 = F.relu(self.conv3_1(p2))
        c3_2 = F.relu(self.conv3_2(c3_1))
        c3_3 = F.relu(self.conv3_3(c3_2))
        p3 = F.average_pooling(c3_3, 2, 2)
        c4_1 = F.relu(self.conv4_1(p3))
        c4_2 = F.relu(self.conv4_2(c4_1))
        c4_3 = F.relu(self.conv4_3(c4_2))
        p4 = F.average_pooling(c4_3, 2, 2)
        c5_1 = F.relu(self.conv5_1(p4))
        c5_2 = F.relu(self.conv5_2(c5_1))
        c5_3 = F.relu(self.conv5_3(c5_2))
        return {'conv1_1':c1_1, 'conv1_2':c1_2, 'conv2_1':c2_1, 'conv2_2':c2_2,
                'conv3_1':c3_1, 'conv3_2':c3_2, 'conv3_3':c3_3, 'conv4_1':c4_1,
                'conv5_1':c5_1, 'conv5_2':c5_2, 'conv5_3':c5_3}

# Setup for content & style image
content_path = dezero.utils.get_file(content_url)
style_path = dezero.utils.get_file(style_url)
content_img = Image.open(content_path)
content_size = content_img.size
style_img = Image.open(style_path)
content_img = VGG16.preprocess(content_img, size=model_input_size)[np.newaxis]  # preprocess for VGG
style_img = VGG16.preprocess(style_img, size=model_input_size)[np.newaxis]
content_img, style_img = Variable(content_img), Variable(style_img)

model = VGG16(pretrained=True)
#gen_data = np.random.uniform(-20, 20, (1, 3, img_resize[0], img_resize[1])).astype(np.float32)
gen_data = content_img.data.copy()
gen_img = dezero.Parameter(gen_data)
gen_model = dezero.models.Model()
gen_model.param = gen_img
optimizer = dezero.optimizers.AdaGrad(lr=lr).setup(gen_model)

if use_gpu:
    model.to_gpu()
    gen_img.to_gpu()
    content_img.to_gpu()
    style_img.to_gpu()


with dezero.no_grad():
    content_features = model.extract(content_img)
    style_features = model.extract(style_img)


def deprocess_image(x, size=None):
    if use_gpu:
        x = dezero.cuda.as_numpy(x)
    if x.ndim == 4:
        x = np.squeeze(x)
    x = x.transpose((1,2,0))
    x += np.array([103.939, 116.779, 123.68])
    x = x[:,:,::-1] # BGR -> RGB
    x = np.clip(x, 0, 255).astype('uint8')
    img = Image.fromarray(x, mode="RGB")
    if size:
        img = img.resize(size)
    return img


def gram_mat(x):
    N, C, H, W = x.shape
    features = x.reshape(C, -1)
    gram = F.matmul(features, features.T)
    return gram.reshape(1, C, C)


def style_loss(style, comb):
    S = gram_mat(style)
    C = gram_mat(comb)
    N, ch, H, W = style.shape
    return F.mean_squared_error(S, C) / (4 * (ch * W * H)**2)


def content_loss(base, comb):
    return F.mean_squared_error(base, comb) / 2


def total_varitaion_loss(x):
    a = (x[:, :, :-1, :-1] - x[:, :, 1:, : -1]) ** 2
    b = (x[:, :, :-1, :-1] - x[:, :, : -1, 1:]) ** 2
    return F.sum(a + b)


def loss_func(gen_features, content_features, style_features, gen_img):
    loss = 0
    # content loss
    for layer in content_features:
        loss += content_weight / len(content_layers) * \
                content_loss(content_features[layer], gen_features[layer])
    # style loss
    for layer in style_features:
        loss += style_weight / len(style_layers) * \
                style_loss(style_features[layer], gen_features[layer])
    # total variation loss
    loss += total_varitaion_weight * total_varitaion_loss(gen_img)
    return loss


print_interval = 100 if use_gpu else 1
for i in range(iterations):
    model.cleargrads()
    gen_img.cleargrad()

    gen_features = model.extract(gen_img)
    loss = loss_func(gen_features, content_features, style_features, gen_img)
    loss.backward()
    optimizer.update()

    if i % print_interval == 0:
        print('{} loss: {:.0f}'.format(i, float(loss.data)))

    if i % 100 == 0:
        img = deprocess_image(gen_img.data, content_size)
        plt.imshow(np.array(img))
        plt.show()
        #img.save("style_transfer_{}.png".format(str(i)))