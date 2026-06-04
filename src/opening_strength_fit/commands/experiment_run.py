from opening_strength_fit.training import train_from_args
from opening_strength_fit.training_args import build_training_parser


def main() -> None:
    parser = build_training_parser("Train an opening-strength short-horizon baseline.")
    args = parser.parse_args()
    train_from_args(args)


if __name__ == "__main__":
    main()
