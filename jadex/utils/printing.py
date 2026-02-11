# Primary Colors
RED = "\033[31m"  # Standard red
GREEN = "\033[32m"  # Standard green
BLUE = "\033[34m"  # Standard blue
YELLOW = "\033[33m"  # Standard yellow
PURPLE = "\033[35m"  # Standard purple

# Refined & Extended Colors
CRIMSON = "\033[38;5;196m"  # Deep red
CORAL = "\033[38;5;209m"  # Orange-red
ORANGE = "\033[38;5;208m"  # True orange
GOLD = "\033[38;5;220m"  # Warm gold
EMERALD = "\033[38;5;46m"  # Vibrant green
LIME = "\033[38;5;10m"  # Bright lime green
TURQUOISE = "\033[38;5;45m"  # Blue-green transition
TEAL = "\033[38;5;6m"  # Deep cyan-teal
ROYAL_BLUE = "\033[38;5;27m"  # Deep, elegant blue
AMETHYST = "\033[38;5;93m"  # Stylish purple
LAVENDER = "\033[38;5;147m"  # Soft pastel purple
ROSE = "\033[38;5;204m"  # Soft pink

# Utility Colors
SILVER = "\033[90m"  # Dim gray for subtle emphasis


RESET = "\033[0m"


def _print_color(message: str, color: str, **kwargs):
    """Helper function to print messages in color."""
    print(f"{color}{message}{RESET}", **kwargs)


def print_green(msg, **kwargs):
    return _print_color(msg, GREEN, **kwargs)


def print_blue(msg, **kwargs):
    return _print_color(msg, BLUE, **kwargs)


def print_red(msg, **kwargs):
    return _print_color(msg, RED, **kwargs)


def print_yellow(msg, **kwargs):
    return _print_color(msg, YELLOW, **kwargs)


def print_purple(msg, **kwargs):
    return _print_color(msg, PURPLE, **kwargs)


COLORS = [ROSE, CORAL, GOLD, LIME, TEAL, LAVENDER]


# Track the current color index
_color_index = 4

_call_counter = {}
_header_printed = set()
_footer_printed = set()


def print_jit(
    info_str: str,
    shape: tuple,
    model_print_info: dict = {},
    header: bool = False,
    footer: bool = False,
    input: bool = False,
    output: bool = False,
    n: int = 1,
):
    import jax

    global _color_index

    if jax.process_index() == 0:
        # Determine current cycle color
        cycle_color = COLORS[_color_index % len(COLORS)]

        # Create unique key from model UUID and info string
        model_uuid = model_print_info.get("uuid", "")
        key = f"{model_uuid}_{info_str}"

        # Initialize or increment counter for this key
        _call_counter[key] = _call_counter.get(key, 0) + 1

        # Print on the n'th call
        if _call_counter[key] == n:
            # Only print header if we haven't printed one for this model yet
            if header and model_uuid not in _header_printed:
                _print_color("=" * 100, cycle_color)
                _header_printed.add(model_uuid)

            # Add INPUT/OUTPUT before "shape"
            io_label = "INPUT " if input else "OUTPUT " if output else ""
            full_message = f"{info_str} {io_label}shape {shape}"
            if model_print_info.get("name"):
                full_message = f"{model_print_info['name']}: {full_message}"

            _print_color(full_message, cycle_color)

            # Only print footer if we haven't printed one for this model yet
            if footer and model_uuid not in _footer_printed:
                _print_color("=" * 100, cycle_color, end="\n\n")
                _footer_printed.add(model_uuid)

                # Move to the next color
                _color_index = (_color_index + 1) % len(COLORS)


_str_call_counter = {}


def print_jit_warning(msg: str, n: int = 1):
    import jax

    if jax.process_index() == 0:
        # Create unique key for this message
        key = f"str_{msg}"

        # Initialize or increment counter for this key
        _str_call_counter[key] = _str_call_counter.get(key, 0) + 1

        # Print on the n'th call
        if _str_call_counter[key] == n:
            print(msg)


def print_jit_str(msg: str, n: int = 1, with_header_footer: bool = False, cycle=True):
    import jax

    global _color_index

    if jax.process_index() == 0:
        # Determine current cycle color
        cycle_color = COLORS[_color_index % len(COLORS)]

        key = f"str_{msg}"
        _str_call_counter[key] = _str_call_counter.get(key, 0) + 1
        # Print on the n'th call
        if _str_call_counter[key] == n:
            if with_header_footer:
                _print_color("=" * 100, cycle_color)
            _print_color(msg, cycle_color)
            if with_header_footer:
                _print_color("=" * 100, cycle_color, end="\n\n")

        # Move to the next color
        if cycle:
            _color_index = (_color_index + 1) % len(COLORS)
