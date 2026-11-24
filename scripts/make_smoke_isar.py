import _bootstrap
import argparse
from isar.synthetic import create_synthetic

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Create explicitly labelled Synthetic/Smoke complex MAT data')
    p.add_argument('--output-root', default='local_validation')
    print(create_synthetic(p.parse_args().output_root))
