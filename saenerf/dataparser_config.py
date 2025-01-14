
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from saenerf.data.saenerf_dataparser import SaENeRFDataParserConfig
from saenerf.data.eds_dataparser import EDSDataparserConfig

saenerf_dataparser = DataParserSpecification(config=SaENeRFDataParserConfig())

eds_dataparser = DataParserSpecification(config=EDSDataparserConfig())