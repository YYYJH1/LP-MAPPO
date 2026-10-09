from dataclasses import dataclass
from pathlib import Path
import yaml

CONFIG = Path(__file__).resolve().parents[1] / 'config'

@dataclass(frozen=True)
class Params:
    data: dict

    @classmethod
    def load(cls, scenario='small'):
        path = Path(scenario)
        if not path.is_file():
            path = CONFIG / 'scenario' / f'{scenario}.yaml'
        data = yaml.safe_load(path.read_text())
        if data['T'] <= 0 or data['delta'] <= 0 or data['T'] % data['delta']:
            raise ValueError('T must be a positive integer multiple of delta')
        if data['w_corr'] != data['w_node']:
            raise ValueError('the scenario requires w_corr = w_node')
        if not 0 < data['w_srv'] < data['w_node'] - 2*data['clearance']:
            raise ValueError('service region must fit the eroded node')
        return cls(data)

    @property
    def N(self):
        return round(self.data['T'] / self.data['delta'])

    def __getitem__(self, key):
        return self.data[key]


def checker_config():
    return yaml.safe_load((CONFIG / 'checker.yaml').read_text())
