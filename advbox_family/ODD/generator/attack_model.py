import sys
import argparse
from pathlib import Path
from typing import Any

d = Path(__file__).resolve().parents[1]
sys.path.append(str(d))

# pylint: disable=wrong-import-position
# choose attack method
from generator.attack_method.eot_attack import EOTAttack

# choose white-box models
from generator.object_detectors.tiny_yolo.tiny_yolo import TinyYOLO
from generator.object_detectors.tiny_yolo_v3.models import TinyYoloV3
from generator.object_detectors.tiny_yolo_v4.yolov4 import TinyYoloV4


# pylint: enable=wrong-import-position


def main(args):
    """
    Executes the algorithm
    :param args:
    :return:
    """
    model = fetch_model(args)

    attack = EOTAttack(model)
    attack(args)


def fetch_model(args) -> Any:
    """
    Fetches model based on the passed arguments
    :param args:
    :return:
    """
    model = args.model

    if model == 'tiny_yolo':
        return TinyYOLO

    if model == 'tiny_yolo_v3':
        return TinyYoloV3

    return TinyYoloV4


def str2bool(value: str) -> bool:
    """
    Turns string literal into boolean for argument parsing

    :param value:
    :return:
    """
    if value.lower() == 'true':
        return True

    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()

    required = parser.add_argument_group('required arguments')
    required.add_argument('--model', help='Choose a model to fine-tune the attack on',
                          choices=('tiny_yolo', 'tiny_yolo_v3', 'tiny_yolo_v4'),
                          type=str, required=True)
    required.add_argument('--input_images', type=str, required=True,
                          help='Path to folder with images to train the network on')
    required.add_argument('--mask_file', type=str, required=True,
                          help='Path to mask that super-imposes the logo on the picture')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--input_logo', help='Path to logo to build the adversarial image from',
                          type=str)
    optional.add_argument('--steps', help='Number of iterations', type=int)
    optional.add_argument('--punishment', help='Hyper-parameter to tune the attack', type=float)
    optional.add_argument('--smoothness_punishment', help='Hyper-parameter to tune the attack',
                          type=float)
    optional.add_argument('--learning_rate', help='Hyper-parameter to tune the attack',
                          type=float)
    optional.add_argument('--skip_pdb', help='De-activates pdb', choices=(True, False),
                          type=str2bool)
    optional.add_argument('--minimise', type=str2bool, choices=(True, False),
                          help='Whether to run a minimisation or maximization problem')
    optional.add_argument('--num_batches', help='Number of batches to split the EOT transform',
                          type=int)

    arguments = parser.parse_args()
    main(arguments)
