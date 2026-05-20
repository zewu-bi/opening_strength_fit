import _bootstrap  # noqa: F401
from opening_strength_fit.training import build_training_parser, train_from_args


def main() -> None:
    parser = build_training_parser(
        "Train an opening-strength short-horizon baseline."
    )
    args = parser.parse_args()
    train_from_args(args)


if __name__ == "__main__":
    main()

