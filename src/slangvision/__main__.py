"""python -m slangvision [setup | undo]

With no argument, report how SlangVision finds NVRTC. `setup` links the NVRTC
installed with PyTorch into SlangPy's package folder (once per environment;
see the README's Install section); `undo` removes that link.
"""

from __future__ import annotations

import sys

from . import _cuda


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    command = args[0] if args else "status"
    try:
        if command == "setup":
            lines = _cuda.setup()
        elif command == "undo":
            lines = _cuda.undo()
        elif command == "status":
            lines = [_cuda.describe() + "."]
            try:
                used = _cuda.configure()
                lines.append(f"SlangVision will use: {used or 'your own CUDA setup'}.")
            except _cuda.SetupError as e:
                lines.append(f"Not ready: {e}.")
        else:
            print(__doc__.strip(), file=sys.stderr)
            return 2
    except (_cuda.SetupError, OSError) as e:
        print(f"slangvision: {e}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
