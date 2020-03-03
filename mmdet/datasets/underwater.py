from mmdet.core import eval_map, eval_recalls
from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class UndeWaterDataset(XMLDataset):

    CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')

    def __init__(self, **kwargs):
        super(UndeWaterDataset, self).__init__(**kwargs)
