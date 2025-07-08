from torch import nn
from compressai.layers import GDN
from compressai.models.utils import conv, deconv, CondConv, CondDeconv


class GoogleAnalysisTransform(nn.Sequential):

    def __init__(self, in_channels, num_features, num_filters, kernel_size, downsample_8=False):
        super(GoogleAnalysisTransform, self).__init__(
            conv(in_channels, num_filters, kernel_size, stride=2),
            GDN(num_filters),
            conv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters),
            conv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters),
            conv(num_filters, num_features, kernel_size, stride=2 if not downsample_8 else 1)
        )

class GoogleSynthesisTransform(nn.Sequential):

    def __init__(self, out_channels, num_features, num_filters, kernel_size, downsample_8=False):
        super(GoogleSynthesisTransform, self).__init__(
            deconv(num_features, num_filters, kernel_size, stride=2 if not downsample_8 else 1),
            GDN(num_filters, inverse=True),
            deconv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters, inverse=True),
            deconv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters, inverse=True),
            deconv(num_filters, out_channels, kernel_size, stride=2)
        )
