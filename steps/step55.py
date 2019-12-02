def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


h, w = 4, 4  # input_shape
kh, kw = 3, 3  # kernel_shape
sh, sw = 1, 1  # stride（縦方向のストライド、横方向のストライド）
ph, pw = 1, 1  # padding（縦方向のパディング、横方向のパディング）

oh = get_conv_outsize(h, kh, sh, ph)
ow = get_conv_outsize(w, kw, sw, pw)
print(oh, ow)