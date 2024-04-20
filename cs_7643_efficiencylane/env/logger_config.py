import logging
import warnings

logging.getLogger('git.cmd').setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("adapters").setLevel(logging.WARNING)
logging.getLogger('accelerate.utils.other').setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module='transformers.utils.generic')
warnings.filterwarnings("ignore", category=Warning, module='accelerate.utils.other')

# # Disable debug messages from specific packages
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ColoredFormatter(logging.Formatter):
    """A custom formatter to add color to log messages."""
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    COLORS = {
        'WARNING': YELLOW,
        'INFO': GREEN,
        'DEBUG': BLUE,
        'CRITICAL': RED,
        'ERROR': RED
    }

    def format(self, record):
        """Format the log message with color."""
        color = self.COLORS.get(record.levelname, self.WHITE)
        prefix = '\033[1;%dm' % (30 + color)
        postfix = '\033[0m'
        record.msg = prefix + str(record.msg) + postfix
        return super().format(record)

def get_logger():
    """Get the configured logger."""
    logger = logging.getLogger(__name__)
    logger.propagate = False
    if not logger.handlers:
        # Configure the logger with a custom formatter
        logging.basicConfig(level=logging.DEBUG)

        # Add color formatter
        color_formatter = ColoredFormatter('%(levelname)s: %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

    return logger
