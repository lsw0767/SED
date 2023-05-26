BATCH_NORM = 2
NUM_CLASSES = 41
def cmpx_conv(kernel, in_ch, out_ch):
    return kernel*kernel*in_ch*out_ch

def cmpx_vggblock(kernel, in_ch, out_ch):
    params = 0
    params += cmpx_conv(kernel, in_ch, out_ch)
    params += cmpx_conv(kernel, out_ch, out_ch)
    params += BATCH_NORM*out_ch
    return params


total_parameters = 0

total_parameters += cmpx_vggblock(3, 1, 32)
total_parameters += cmpx_vggblock(3, 32, 64)
total_parameters += cmpx_vggblock(3, 64, 128)
total_parameters += cmpx_vggblock(3, 128, 128)

total_parameters += cmpx_conv(1, 128, NUM_CLASSES)
print('total parameters of original model: ', total_parameters/1e3, 'K')

# total_parameters += cmpx_conv(1, 32, NUM_CLASSES)
# total_parameters += cmpx_conv(1, 64, NUM_CLASSES)
# total_parameters += cmpx_conv(1, 128, NUM_CLASSES)
# print('total parameters of MH model:', total_parameters/1e3, 'K')


total_parameters = 0

total_parameters += cmpx_vggblock(3, 1, 16)
total_parameters += cmpx_vggblock(3, 16, 32)
total_parameters += cmpx_vggblock(3, 32, 64)
total_parameters += cmpx_vggblock(3, 64, 128)
total_parameters += cmpx_vggblock(3, 128, 64)
total_parameters += cmpx_vggblock(3, 64, 32)
total_parameters += cmpx_vggblock(3, 32, 16)

total_parameters += cmpx_conv(1, 16, NUM_CLASSES)
print('total parameters of small_SU-Net: ', total_parameters/1e3, 'K')
